import logging
import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import config, db
from ..auth import AuthContext, AuthError, authorized, ensure_owner
from ..clustering import (
    SUGGESTION_INDEX,
    best_match,
    build_suggestion_index,
    build_similarity_groups,
    iou,
    merge_centroid,
)
from ..metrics import reader_requests

router = APIRouter()

logger = logging.getLogger("faces-service")

MAX_CROP_BYTES = 5 * 1024 * 1024
LIST_FACES_DEFAULT_LIMIT = 50
LIST_FACES_MAX_LIMIT = 100


class RenameBody(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class MergeBody(BaseModel):
    cluster_ids: list[str] = Field(min_length=2, max_length=100)


class BatchDeleteBody(BaseModel):
    cluster_ids: list[str] = Field(min_length=1, max_length=100)


def _auth(authorization: str | None = Header(None)) -> AuthContext:
    return authorized(authorization)


@router.get("/faces")
def list_faces(
    auth: AuthContext = Depends(_auth),
    limit: int = Query(default=LIST_FACES_DEFAULT_LIMIT, ge=1, le=LIST_FACES_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
):
    """People list. Clusters with 0 videos (orphans) are hidden and counted in
    ``empty_count``; the rest (``clusters``) is paginated by samples. Similarity
    groups (near-duplicates by centroid cosine >= 0.5) are computed server-side
    and returned in full on every page; grouped clusters are excluded from the
    flat list. ``total`` is the number of video-bearing clusters, ``rest_total``
    the number of non-grouped ones (the pagination bound). Centroid vectors are
    stripped from list responses to keep the payload small."""
    owner_id = auth.user_id
    clusters = db.get_clusters_for_owner(owner_id)
    videos_by_cluster = db.get_clusters_videos(owner_id)

    def _public(c):
        d = dict(c)
        d.pop("centroid", None)
        return d

    # keep only clusters that appear in at least one video
    rest = [c for c in clusters if c["id"] in videos_by_cluster]
    empty_count = len(clusters) - len(rest)

    all_groups = build_similarity_groups([c for c in rest if c.get("centroid")])
    grouped_ids = {
        g["rep"]["id"] for g in all_groups
    } | {m["cluster"]["id"] for g in all_groups for m in g["members"]}

    flat = [c for c in rest if c["id"] not in grouped_ids]
    flat.sort(
        key=lambda c: (
            not c["is_named"],
            c["name"] is None,
            c["name"] or "",
            -c["sample_count"],
            c["id"],
        )
    )

    total = len(rest)
    rest_total = len(flat)
    page = flat[offset : offset + limit]

    def stats(c):
        videos = videos_by_cluster.get(c["id"], [])
        return {
            **_public(c),
            "video_count": len(videos),
            "videos": videos,
        }

    groups_out = []
    for g in all_groups:
        groups_out.append(
            {
                "rep": stats(g["rep"]),
                "members": [
                    {"cluster": stats(m["cluster"]), "sim": m["sim"]}
                    for m in g["members"]
                ],
                "maxSim": g["maxSim"],
            }
        )

    reader_requests.labels(endpoint="list_faces", status="200").inc()
    return {
        "clusters": [stats(c) for c in page],
        "groups": groups_out,
        "total": total,
        "empty_count": empty_count,
        "rest_total": rest_total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/faces/delete-empty")
def delete_empty_faces(request: Request, auth: AuthContext = Depends(_auth)):
    """Permanently delete clusters that appear in no video (0 samples), purging
    their crop images too. Idempotent: clusters that gained occurrences between
    the query and the delete are skipped. Returns the number actually deleted.
    streams' faces_detected flag is untouched — empty clusters have no
    occurrences, so no stream needs a detection reset."""
    rows = db.get_clusters_without_occurrences(auth.user_id)
    deleted = 0
    store = getattr(request.app.state, "store", None)
    for r in rows:
        result = db.delete_cluster(r["id"], auth.user_id)
        if result is None:
            continue
        deleted += 1
        if store is not None and result.get("crop_object"):
            try:
                store.delete_crop(result["owner_id"], r["id"])
            except Exception as e:
                logger.warning("crop delete error cluster=%s: %s", r["id"], e)
    reader_requests.labels(endpoint="delete_empty_faces", status="200").inc()
    return {"deleted": deleted}


@router.post("/faces/delete-batch")
def delete_faces_batch(
    body: BatchDeleteBody,
    request: Request,
    auth: AuthContext = Depends(_auth),
):
    """Permanently delete the given face clusters (their occurrences are removed
    via cascade). Idempotent via dedupe: unknown/foreign cluster ids are skipped
    silently. Returns the number actually deleted. streams' faces_detected flag
    is untouched — deleting persons is curation of an already-run detection."""
    deleted = 0
    store = getattr(request.app.state, "store", None)
    for cid in dict.fromkeys(body.cluster_ids):
        result = db.delete_cluster(cid, auth.user_id)
        if result is None:
            continue
        deleted += 1
        if store is not None and result.get("crop_object"):
            try:
                store.delete_crop(result["owner_id"], cid)
            except Exception as e:
                logger.warning("crop delete error cluster=%s: %s", cid, e)
    reader_requests.labels(endpoint="delete_batch_faces", status="200").inc()
    return {"deleted": deleted}


@router.get("/faces/suggest")
def suggest_faces(
    auth: AuthContext = Depends(_auth),
    q: str = Query(min_length=3, max_length=120),
    exclude: str | None = Query(default=None, max_length=120),
):
    """Name-prefix suggestions for the rename editor: named clusters of the
    owner whose name starts with ``q`` (case-insensitive, literal). The person
    currently being renamed can be excluded by id. Lightweight payload (no
    centroid/videos)."""
    owner_id = auth.user_id
    clusters = db.search_named_clusters(owner_id, q.strip())
    if exclude:
        clusters = [c for c in clusters if c["id"] != exclude]
    reader_requests.labels(endpoint="suggest", status="200").inc()
    return {"clusters": clusters}


# The interactive frame assist runs insightface inside the reader process, whose
# thread pool is shared with every other reader endpoint. Without a cap a paused
# player (or a scrubbing user) can pile up CPU-heavy inference runs.
_INFER_SEMAPHORE = threading.BoundedSemaphore(2)


def _engine_or_503(request: Request, endpoint: str):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        reader_requests.labels(endpoint=endpoint, status="503").inc()
        raise HTTPException(status_code=503, detail="engine not ready")
    return engine


def _public_cluster(c: dict) -> dict:
    """Cluster without the 512-float centroid / videos — the overlay only needs
    enough to render a name and to attach on confirm."""
    return {
        "id": str(c["id"]),
        "name": c.get("name"),
        "is_named": bool(c.get("is_named")),
        "sample_count": int(c.get("sample_count") or 0),
        "crop_object": c.get("crop_object"),
    }


async def _read_image(request: Request, file: UploadFile, endpoint: str) -> np.ndarray:
    raw = await file.read(MAX_CROP_BYTES + 1)
    if len(raw) > MAX_CROP_BYTES:
        reader_requests.labels(endpoint=endpoint, status="413").inc()
        raise HTTPException(status_code=413, detail="file too large")
    img = _decode_image(raw)
    if img is None:
        reader_requests.labels(endpoint=endpoint, status="400").inc()
        raise HTTPException(status_code=400, detail="invalid image")
    return img


@router.post("/faces/detect")
async def detect_faces(
    request: Request,
    file: UploadFile = File(...),
    auth: AuthContext = Depends(_auth),
):
    """Detect the faces in a single frame the owner paused on and offer a
    cluster per face.

    The client grabs the paused frame from the <video> element (canvas -> JPEG)
    and posts it here; we run the same detector/recognizer the batch pipeline
    uses, then match every face against the owner's clusters. A face whose best
    similarity is below `unknown_threshold` gets no suggestion at all — the
    client then offers to create a new person instead.
    """
    started = time.perf_counter()
    img = await _read_image(request, file, "detect_frame")
    engine = _engine_or_503(request, "detect_frame")

    owner_id = auth.user_id
    auto_threshold = config.Config.assist_auto_threshold
    floor = config.Config.unknown_threshold

    # The centroid index is cached per owner: loading every cluster on every
    # pause cost ~2s for a library of a few thousand people, which dominated the
    # whole request. Writes drop the owner's entry, so this is a fresh read.
    matrix, meta = SUGGESTION_INDEX.get(
        owner_id, lambda: build_suggestion_index(db.get_clusters_for_owner(owner_id))
    )
    index_ms = (time.perf_counter() - started) * 1000

    with _INFER_SEMAPHORE:
        detections = engine.model.embed(
            img,
            config.Config.detect_threshold,
            config.Config.assist_det_size,
            config.Config.assist_min_face_ratio,
        )
    infer_ms = (time.perf_counter() - started) * 1000 - index_ms

    height, width = img.shape[:2]
    faces = []
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        cluster, sim = best_match(matrix, meta, det["embedding"], floor)
        suggestion = None
        if cluster is not None:
            suggestion = {
                **_public_cluster(cluster),
                "similarity": round(float(sim), 4),
                # strong enough to attach without asking the user
                "auto": bool(sim >= auto_threshold),
            }
        faces.append(
            {
                "bbox": [float(v) for v in bbox],
                "confidence": float(det["confidence"]),
                "suggestion": suggestion,
            }
        )

    took_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "detect frame faces=%d clusters=%d det=%d index=%.0fms infer=%.0fms total=%.0fms",
        len(faces),
        len(meta),
        config.Config.assist_det_size,
        index_ms,
        infer_ms,
        took_ms,
    )
    reader_requests.labels(endpoint="detect_frame", status="200").inc()
    return {
        "faces": faces,
        "width": int(width),
        "height": int(height),
        "took_ms": round(took_ms, 1),
    }


@router.post("/streams/{stream_id}/faces/attach")
async def attach_frame_face(
    stream_id: str,
    request: Request,
    file: UploadFile = File(...),
    bbox: str = Form(...),
    t_seconds: float = Form(...),
    cluster_id: str | None = Form(default=None),
    auth: AuthContext = Depends(_auth),
):
    """Attach the paused frame's face to a cluster (owner/admin only).

    Takes the whole frame plus the bbox the client drew, not a pre-cropped face:
    the detector is re-run server-side and the embedding is taken from the box
    that matches, so the client never supplies its own vector (and a tight crop
    that the detector cannot re-detect still works).

    `cluster_id` empty -> create a new anonymous cluster first, exactly like the
    batch pipeline does. The occurrence is deduped by
    (stream_id, cluster_id, t_seconds), so re-pausing on the same second is a
    no-op rather than a duplicate sample.
    """
    img = await _read_image(request, file, "attach_frame_face")
    engine = _engine_or_503(request, "attach_frame_face")

    try:
        box = [float(v) for v in bbox.split(",")]
    except ValueError:
        reader_requests.labels(endpoint="attach_frame_face", status="400").inc()
        raise HTTPException(status_code=400, detail="bbox must be x1,y1,x2,y2")
    if len(box) != 4 or not all(v == v for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        reader_requests.labels(endpoint="attach_frame_face", status="400").inc()
        raise HTTPException(status_code=400, detail="bbox must be x1,y1,x2,y2")
    if not (t_seconds >= 0) or t_seconds != t_seconds or t_seconds == float("inf"):
        reader_requests.labels(endpoint="attach_frame_face", status="400").inc()
        raise HTTPException(status_code=400, detail="t_seconds out of range")

    created = False
    if cluster_id:
        cluster = db.get_cluster(cluster_id)
        if cluster is None:
            reader_requests.labels(endpoint="attach_frame_face", status="404").inc()
            raise HTTPException(status_code=404, detail="not found")
        try:
            ensure_owner(auth, str(cluster["owner_id"]))
        except AuthError as e:
            reader_requests.labels(endpoint="attach_frame_face", status=str(e.status)).inc()
            raise HTTPException(status_code=e.status, detail=e.message)
    else:
        cluster = None

    with _INFER_SEMAPHORE:
        # same detector size and size floor as /faces/detect, so the box the
        # client echoes back was produced at the same resolution we re-detect at
        detections = engine.model.embed(
            img,
            config.Config.detect_threshold,
            config.Config.assist_det_size,
            config.Config.assist_min_face_ratio,
        )

    target = None
    best_iou = 0.0
    for det in detections:
        det_box = det.get("bbox")
        if not det_box:
            continue
        score = iou(det_box, box)
        if score > best_iou:
            best_iou = score
            target = det
    if target is None or best_iou < 0.5:
        reader_requests.labels(endpoint="attach_frame_face", status="409").inc()
        raise HTTPException(status_code=409, detail="face not found in the given box")

    embedding = target["embedding"]
    centroid = np.asarray(embedding, dtype=np.float32)
    owner_id = auth.user_id if cluster is None else str(cluster["owner_id"])

    if cluster is None:
        new_id = db.upsert_cluster(owner_id, None, False, centroid.tolist())
        created = True
        cluster = db.get_cluster(new_id, owner_id)
    cluster_id = str(cluster["id"])

    written = db.append_stream_occurrences(
        owner_id,
        stream_id,
        [(cluster_id, centroid.tolist(), t_seconds, float(target["confidence"]))],
    )
    # keep the identity centroid fresh, same running mean as the batch path
    previous = np.asarray(cluster["centroid"], dtype=np.float32)
    merged = merge_centroid(previous, int(cluster["sample_count"] or 0), centroid)
    db.update_cluster_centroid(cluster_id, owner_id, merged.tolist())

    if not cluster.get("crop_object") and target.get("bbox"):
        jpeg = engine._crop_jpeg(img, list(target["bbox"]))
        store = getattr(request.app.state, "store", None)
        if jpeg is not None and store is not None:
            key = store.put_crop(owner_id, cluster_id, jpeg)
            db.set_cluster_crop(cluster_id, owner_id, key)

    updated = db.get_cluster(cluster_id, owner_id)
    reader_requests.labels(endpoint="attach_frame_face", status="200").inc()
    return {
        "cluster": updated,
        "created": created,
        "written": written,
        "t_seconds": t_seconds,
    }


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
    return Response(
        content=content,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


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


@router.delete("/streams/{stream_id}/faces/{cluster_id}")
def detach_stream_face(
    stream_id: str,
    cluster_id: str,
    request: Request,
    auth: AuthContext = Depends(_auth),
):
    """Remove the association between a person and one of their videos: the
    cluster's occurrences in this stream are deleted. If the cluster then has
    no occurrences in any stream it is deleted entirely (along with its crop).
    The faces_detected flag is intentionally left unchanged: this is curation
    (like delete/merge), not a signal to re-run detection."""
    c = db.get_cluster(cluster_id)
    if c is None:
        reader_requests.labels(endpoint="detach_stream_face", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")
    try:
        ensure_owner(auth, c["owner_id"])
    except AuthError as e:
        reader_requests.labels(endpoint="detach_stream_face", status=str(e.status)).inc()
        raise HTTPException(status_code=e.status, detail=e.message)

    result = db.detach_cluster_from_stream(cluster_id, stream_id, c["owner_id"])
    if result is None:
        reader_requests.labels(endpoint="detach_stream_face", status="404").inc()
        raise HTTPException(status_code=404, detail="not found")

    store = getattr(request.app.state, "store", None)
    if store is not None and result.get("crop_object"):
        try:
            store.delete_crop(c["owner_id"], cluster_id)
        except Exception as e:
            logger.warning("crop delete during detach error cluster=%s: %s", cluster_id, e)

    reader_requests.labels(endpoint="detach_stream_face", status="200").inc()
    return {
        "cluster_id": cluster_id,
        "stream_id": stream_id,
        "removed": result["removed"],
        "sample_count": result["sample_count"],
        "cluster_deleted": result["cluster_deleted"],
    }


@router.delete("/faces/{cluster_id}")
def delete_face(cluster_id: str, request: Request, auth: AuthContext = Depends(_auth)):
    """Delete a person (owner/admin only). Removes the cluster, its occurrences
    and its crop image. The faces_detected flag is intentionally left unchanged:
    detection already ran, and deleting a person is curation (unwanted or broken
    detections), not a signal that detection needs to run again."""
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