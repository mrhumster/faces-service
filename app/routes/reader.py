import io

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import db
from ..auth import AuthContext, AuthError, authorized, ensure_owner
from ..metrics import reader_requests

router = APIRouter()

MAX_CROP_BYTES = 5 * 1024 * 1024


class RenameBody(BaseModel):
    name: str | None = Field(default=None, max_length=120)


def _auth(authorization: str | None = Header(None)) -> AuthContext:
    return authorized(authorization)


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