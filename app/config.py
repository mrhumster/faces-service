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