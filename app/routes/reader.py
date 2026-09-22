from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..auth import AuthContext, AuthError, authorized, ensure_owner
from ..metrics import reader_requests

router = APIRouter()


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
    name = body.get("name")
    err = db.rename_cluster(cluster_id, c["owner_id"], name)
    if err is not None:
        code, msg = err
        status = 404 if code == "not_found" else 409
        reader_requests.labels(endpoint="rename_face", status=str(status)).inc()
        raise HTTPException(status_code=status, detail=msg)
    reader_requests.labels(endpoint="rename_face", status="200").inc()
    return db.get_cluster(cluster_id)


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