from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from .. import config
from ..auth import AuthError, internal_token
from ..metrics import infer_duration, infer_faces_detected, infer_occurrences_written, infer_total

router = APIRouter()


class InferRequest(BaseModel):
    stream_id: str
    owner_id: str
    frames_prefix: str
    count: int


@router.post("/infer")
def infer(body: InferRequest, request: Request, x_internal_token: str | None = Header(None)):
    try:
        internal_token(x_internal_token)
    except AuthError as e:
        infer_total.labels(status=str(e.status)).inc()
        raise HTTPException(status_code=e.status, detail=e.message)

    if not (0 < body.count <= config.Config.max_frames):
        infer_total.labels(status="400").inc()
        raise HTTPException(status_code=400, detail="count out of range")

    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        infer_total.labels(status="503").inc()
        raise HTTPException(status_code=503, detail="engine not ready")

    with infer_duration.time():
        result = engine.run(body.stream_id, body.owner_id, body.frames_prefix, body.count)
    infer_faces_detected.inc(result["faces_detected"])
    infer_occurrences_written.inc(result["occurrences_written"])
    infer_total.labels(status="200").inc()
    return result