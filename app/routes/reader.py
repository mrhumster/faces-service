import io
import logging

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import config, db
from ..auth import AuthContext, AuthError, authorized, ensure_owner
from ..metrics import reader_requests

router = APIRouter()

logger = logging.getLogger("faces-service")

MAX_CROP_BYTES = 5 * 1024 * 1024


class RenameBody(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class MergeBody(BaseModel):
    cluster_ids: list[str] = Field(min_length=2, max_length=100)


def _auth(authorization: str | None = Header(None)) -> AuthContext:
    return authorized(authorization)


def _reset_streams_faces(stream_ids: list[str]) -> None:
    """Best-effort: tell stream-service to clear the faces_detected flag on the
    affected streams so detection can be re-run after a person is deleted.
    Does not raise: stream 404s and upstream errors are logged and skipped."""
    if not config.Config.internal_token:
        return
    if not config.Config.stream_service_url:
        return
    headers = {"X-Internal-Token": config.Config.internal_token}
    for stream_id in stream_ids:
        url = f"{config.Config.stream_service_url.rstrip('/')}/stream/{stream_id}/faces/reset"
        try:
            r = httpx.post(url, headers=headers, timeout=3.0)
            if r.status_code not in (200, 201, 404):
                logger.warning("reset faces flag failed stream=%s status=%s", stream_id, r.status_code)
        except Exception as e:
            logger.warning("reset faces flag error stream=%s: %s", stream_id, e)


@router.get("/faces")
def list_faces(auth: AuthContext = Depends(_auth)):
    owner_id = auth.user_id
    clusters = db.get_clusters_for_owner(owner_id)
    total = db.get_cluster_total_counts(owner_id)

    def stats(c):
        videos = db.get_cluster_videos(c["id"])
        return {
            **c,
            "video_count": len(videos),
            "videos": videos,
        }

    reader_requests.labels(endpoint="list_faces", status="200").inc()
    return {"clusters": [stats(c) for c in clusters], "total": total}


@router.get("/faces/{cluster_id}")
def get_face(cluster_id: str, auth: AuthContext = Depends(_auth)):
    c = db.get_cluster(cluster_id)
    if c is None:
        reader_requests.labels(endpoint="get_face", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")
    try:
        ensure_owner(auth, c["owner_id"])
    except AuthError as e:
        reader_requests.labels(endpoint="get_face", status=str(e.status)).inc()
        raise HTTPException(status_code=e.status, detail=e.message)
    occurrences = db.get_cluster_occurrences(cluster_id)
    videos = db.get_cluster_videos(cluster_id)
    reader_requests.labels(endpoint="get_face", status="200").inc()
    return {
        "cluster": c,
        "occurrences": occurrences,
        "videos": videos,
    }


@router.patch("/faces/{cluster_id}")
def rename_face(cluster_id: str, body: RenameBody, auth: AuthContext = Depends(_auth)):
    c = db.get_cluster(cluster_id)
    if c is None:
        reader_requests.labels(endpoint="rename_face", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")
    try:
        ensure_owner(auth, c["owner_id"])
    except AuthError as e:
        reader_requests.labels(endpoint="rename_face", status=str(e.status)).inc()
        raise HTTPException(status_code=e.status, detail=e.message)
    name = body.name
    err = db.rename_cluster(cluster_id, c["owner_id"], name)
    if err is not None:
        code, msg = err
        status = 404 if code == "not_found" else 409
        reader_requests.labels(endpoint="rename_face", status=str(status)).inc()
        raise HTTPException(status_code=status, detail=msg)
    reader_requests.labels(endpoint="rename_face", status="200").inc()
    return db.get_cluster(cluster_id)


@router.get("/faces/{cluster_id}/crop")
def face_crop(cluster_id: str, request: Request, auth: AuthContext = Depends(_auth)):
    """Serve the cluster face-crop image (owner/admin only)."""
    c = db.get_cluster(cluster_id)
    if c is None:
        reader_requests.labels(endpoint="face_crop", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")
    try:
        ensure_owner(auth, c["owner_id"])
    except AuthError as e:
        reader_requests.labels(endpoint="face_crop", status=str(e.status)).inc()
        raise HTTPException(status_code=e.status, detail=e.message)
    store = getattr(request.app.state, "store", None)
    if store is None:
        reader_requests.labels(endpoint="face_crop", status="503").inc()
        raise HTTPException(status_code=503, detail="storage unavailable")
    data = store.get_crop(c["owner_id"], cluster_id)
    if data is None:
        reader_requests.labels(endpoint="face_crop", status="404").inc()
        raise HTTPException(status_code=404, detail="no crop yet")
    content, content_type = data
    reader_requests.labels(endpoint="face_crop", status="200").inc()
    return Response(content=content, media_type=content_type)


@router.put("/faces/{cluster_id}/crop")
async def replace_face_crop(
    cluster_id: str,
    request: Request,
    file: UploadFile = File(...),
    auth: AuthContext = Depends(_auth),
):
    """Replace the cluster face-crop image (owner/admin only, multipart upload)."""
    c = db.get_cluster(cluster_id)
    if c is None:
        reader_requests.labels(endpoint="replace_face_crop", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")
    try:
        ensure_owner(auth, c["owner_id"])
    except AuthError as e:
        reader_requests.labels(endpoint="replace_face_crop", status=str(e.status)).inc()
        raise HTTPException(status_code=e.status, detail=e.message)

    raw = await file.read(MAX_CROP_BYTES + 1)
    if len(raw) > MAX_CROP_BYTES:
        reader_requests.labels(endpoint="replace_face_crop", status="413").inc()
        raise HTTPException(status_code=413, detail="file too large")
    img = _decode_image(raw)
    if img is None:
        reader_requests.labels(endpoint="replace_face_crop", status="400").inc()
        raise HTTPException(status_code=400, detail="invalid image")

    ok, buf = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        reader_requests.labels(endpoint="replace_face_crop", status="500").inc()
        raise HTTPException(status_code=500, detail="image encode failed")
    jpeg = buf.tobytes()

    store = getattr(request.app.state, "store", None)
    if store is None:
        reader_requests.labels(endpoint="replace_face_crop", status="503").inc()
        raise HTTPException(status_code=503, detail="storage unavailable")
    key = store.put_crop(c["owner_id"], cluster_id, jpeg, content_type="image/jpeg")
    db.set_cluster_crop(cluster_id, c["owner_id"], key)

    updated = db.get_cluster(cluster_id)
    reader_requests.labels(endpoint="replace_face_crop", status="200").inc()
    return {"cluster": updated, "crop_object": key}


def _decode_image(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    return img


@router.get("/streams/{stream_id}/faces")
def stream_faces(stream_id: str, auth: AuthContext = Depends(_auth)):
    """Clusters that appear in a given stream (used by "People in this video")."""
    rows = db.get_occurrences_for_stream(stream_id)
    clusters = {}
    for r in rows:
        clusters.setdefault(r["cluster_id"], []).append(r)
    out = []
    for cluster_id, occs in clusters.items():
        c = db.get_cluster(cluster_id)
        if c is None:
            continue
        try:
            ensure_owner(auth, c["owner_id"])
        except AuthError:
            continue
        out.append(
            {
                "cluster": c,
                "count": len(occs),
                "occurrences": occs,
            }
        )
    reader_requests.labels(endpoint="stream_faces", status="200").inc()
    return {"clusters": out}


@router.delete("/faces/{cluster_id}")
def delete_face(cluster_id: str, request: Request, auth: AuthContext = Depends(_auth)):
    """Delete a person (owner/admin only). Removes the cluster, its occurrences
    and its crop image, then resets the faces_detected flag on every affected
    stream so detection can be re-run."""
    c = db.get_cluster(cluster_id)
    if c is None:
        reader_requests.labels(endpoint="delete_face", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")
    try:
        ensure_owner(auth, c["owner_id"])
    except AuthError as e:
        reader_requests.labels(endpoint="delete_face", status=str(e.status)).inc()
        raise HTTPException(status_code=e.status, detail=e.message)

    result = db.delete_cluster(cluster_id, c["owner_id"])
    if result is None:
        reader_requests.labels(endpoint="delete_face", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")

    store = getattr(request.app.state, "store", None)
    if store is not None and result.get("crop_object"):
        try:
            store.delete_crop(result["owner_id"], cluster_id)
        except Exception as e:
            logger.warning("crop delete error cluster=%s: %s", cluster_id, e)

    _reset_streams_faces(result.get("affected_streams", []))
    reader_requests.labels(endpoint="delete_face", status="200").inc()
    return {"cluster_id": cluster_id}


@router.post("/faces/merge")
def merge_faces(body: MergeBody, request: Request, auth: AuthContext = Depends(_auth)):
    """Merge several people into one. Occurrences move to the target cluster
    (the first selected cluster that has a name, otherwise the first one), the
    source clusters are deleted along with their crops. The faces_detected
    flag is intentionally left unchanged (detection already ran)."""
    ids = body.cluster_ids
    target_id: str | None = None
    target_owner: str | None = None
    for cid in ids:
        c = db.get_cluster(cid)
        if c is None:
            reader_requests.labels(endpoint="merge_faces", status="404").inc()
            raise HTTPException(status_code=404, detail="not found")
        try:
            ensure_owner(auth, c["owner_id"])
        except AuthError as e:
            reader_requests.labels(endpoint="merge_faces", status=str(e.status)).inc()
            raise HTTPException(status_code=e.status, detail=e.message)
        if target_id is None and c["is_named"]:
            target_id = cid
            target_owner = c["owner_id"]
    if target_id is None:
        target_id = ids[0]
        target_owner = db.get_cluster(target_id)["owner_id"]

    source_ids = [cid for cid in ids if cid != target_id]
    result = db.merge_clusters(target_owner, target_id, source_ids)
    if result is None:
        reader_requests.labels(endpoint="merge_faces", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")

    store = getattr(request.app.state, "store", None)
    if store is not None and result.get("deleted_cluster_ids"):
        for cid in result["deleted_cluster_ids"]:
            try:
                store.delete_crop(target_owner, cid)
            except Exception as e:
                logger.warning("crop delete during merge error cluster=%s: %s", cid, e)

    merged = db.get_cluster(target_id, target_owner)
    reader_requests.labels(endpoint="merge_faces", status="200").inc()
    return {"cluster": merged, "merged_ids": ids, "target_id": target_id}