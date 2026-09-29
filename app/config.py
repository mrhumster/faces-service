import os


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, str(default)).strip().lower()
    return raw in ("1", "true", "yes", "on")


class Config:
    server_addr: str = _env("SERVER_ADDR", ":8080")
    mode: str = _env("MODE", "release")

    db_host: str = _env("DB_HOST", "localhost")
    db_port: str = _env("DB_PORT", "5432")
    db_user: str = _env("DB_USER", "postgres")
    db_pass: str = _env("DB_PASS", "")
    db_name: str = _env("DB_NAME", "faces")
    db_max_conn: int = int(_env("DB_MAX_CONN", "40"))
    db_acquire_timeout: float = float(_env("DB_ACQUIRE_TIMEOUT", "5"))
    reader_threads: int = int(_env("READER_THREADS", "80"))

    minio_endpoint: str = _env("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = _env("MINIO_ACCESS_KEY", "admin")
    minio_secret_key: str = _env("MINIO_SECRET_KEY", "minio123")
    minio_bucket: str = _env("MINIO_BUCKET_NAME", "stream-service-test")
    minio_use_ssl: bool = _env_bool("MINIO_USE_SSL", False)

    jwt_public_key_url: str = _env("JWT_ACCESS_PUBLIC_KEY_URL", "")
    cors_origins: str = _env("CORS_ALLOW_ORIGINS", "")

    internal_token: str = _env("FACES_INTERNAL_TOKEN", "")

    stream_service_url: str = _env("STREAM_SERVICE_URL", "http://stream-service:80")

    models_root: str = _env("MODELS_ROOT", "/models")
    match_threshold: float = float(_env("FACES_MATCH_THRESHOLD", "0.4"))
    unknown_threshold: float = float(_env("FACES_UNKNOWN_THRESHOLD", "0.5"))
    max_frames: int = int(_env("FACES_MAX_FRAMES", "500"))
    detect_threshold: float = float(_env("FACES_DETECT_THRESHOLD", "0.4"))
    # Interactive frame assist (owner pauses the player): a suggestion at or above
    # this similarity is trusted enough to attach silently, below it the user is
    # asked. A suggestion floor of unknown_threshold decides "candidate at all".
    assist_auto_threshold: float = float(_env("FACES_ASSIST_AUTO_THRESHOLD", "0.9"))
    # Detector input for the interactive assist. SCRFD work is quadratic in this,
    # and the batch pipeline keeps the full 640 by leaving it at 0.
    assist_det_size: int = int(_env("FACES_ASSIST_DET_SIZE", "320"))
    # Faces shorter than this share of the frame are dropped before recognition
    # (~589ms each, and too small to match anyone). 0 keeps every face.
    assist_min_face_ratio: float = float(_env("FACES_ASSIST_MIN_FACE_RATIO", "0.06"))