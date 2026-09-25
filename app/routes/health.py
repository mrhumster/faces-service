from fastapi import APIRouter

from .. import db
from ..metrics import reader_requests

router = APIRouter()


@router.get("/health")
def health():
    try:
        db.ping_direct()
        return {"status": "up"}
    except Exception:
        return {"status": "down"}