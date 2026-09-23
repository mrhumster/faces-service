import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config, db
from .auth import AuthError, init_token_service
from .infer import InferEngine
from .model import FaceModel
from .minio_client import FrameStore
from .routes import health, metrics, reader, infer as infer_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        db.init_db()
    except Exception as e:
        logging.getLogger("faces-service").warning("db init deferred: %s", e)
    try:
        init_token_service()
    except AuthError as e:
        logging.getLogger("faces-service").warning(
            "token service init deferred: %s", e.message
        )
    model = FaceModel()
    store = FrameStore()
    app.state.store = store
    app.state.engine = InferEngine(model=model, store=store)
    logging.getLogger("faces-service").info(
        "faces-service up addr=%s bucket=%s",
        config.Config.server_addr,
        config.Config.minio_bucket,
    )
    yield
    db.close_db()


def create_app() -> FastAPI:
    app = FastAPI(title="faces-service", lifespan=lifespan, docs_url=None, redoc_url=None)
    origins = [
        o.strip().strip('"')
        for o in config.Config.cors_origins.split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AuthError)
    async def auth_error_handler(request: Request, exc: AuthError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(reader.router)
    app.include_router(infer_router.router)
    return app


app = create_app()