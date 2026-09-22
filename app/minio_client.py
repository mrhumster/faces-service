import io

from minio import Minio

from . import config


class FrameStore:
    def __init__(self) -> None:
        self.client = Minio(
            config.Config.minio_endpoint,
            access_key=config.Config.minio_access_key,
            secret_key=config.Config.minio_secret_key,
            secure=config.Config.minio_use_ssl,
        )
        self.bucket = config.Config.minio_bucket

    def get_frame(self, prefix: str, index: int) -> bytes | None:
        key = f"{prefix.rstrip('/')}/frame_{index:05d}.jpg"
        try:
            resp = self.client.get_object(self.bucket, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()
        except Exception:
            return None

    def get_frame_io(self, prefix: str, index: int) -> tuple[io.BytesIO, int] | None:
        data = self.get_frame(prefix, index)
        if data is None:
            return None
        return io.BytesIO(data), len(data)