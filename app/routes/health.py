from fastapi import APIRouter

from .. import db
from ..metrics import reader_requests

router = APIRouter()


@router.get("/health")
def health():
    try:
        db.ping()
        return {"status": "up"}
    except Exception:
        return {"status": "down"}