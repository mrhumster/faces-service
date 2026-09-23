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

    def delete_prefix(self, prefix: str, recursive: bool = True) -> None:
        """Remove every object under prefix (e.g. "faces/<stream_id>/")."""
        prefix = prefix.rstrip("/") + "/"
        objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=recursive)
        for obj in objects:
            try:
                self.client.remove_object(self.bucket, obj.object_name)
            except Exception as e:
                # best-effort: a missing object is fine, surface nothing else
                if getattr(e, "code", None) not in ("NoSuchKey", "NotFound"):
                    raise

    @staticmethod
    def crop_key(owner_id: str, cluster_id: str) -> str:
        return f"faces/crops/{owner_id}/{cluster_id}.jpg"

    def put_crop(self, owner_id: str, cluster_id: str, data: bytes, content_type: str = "image/jpeg") -> str:
        """Store (or replace) a cluster face crop. Returns the object key."""
        key = self.crop_key(owner_id, cluster_id)
        self.client.put_object(
            self.bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def get_crop(self, owner_id: str, cluster_id: str) -> tuple[bytes, str] | None:
        key = self.crop_key(owner_id, cluster_id)
        try:
            resp = self.client.get_object(self.bucket, key)
            try:
                return resp.read(), resp.headers.get("Content-Type", "image/jpeg")
            finally:
                resp.close()
                resp.release_conn()
        except Exception:
            return None

    def delete_crop(self, owner_id: str, cluster_id: str) -> None:
        try:
            self.client.remove_object(self.bucket, self.crop_key(owner_id, cluster_id))
        except Exception as e:
            if getattr(e, "code", None) not in ("NoSuchKey", "NotFound"):
                raise

    def delete_stream_frames(self, stream_id: str) -> None:
        """Remove sampled frames for a stream (crops are per-cluster and are
        purged separately when a cluster itself is deleted)."""
        self.delete_prefix(f"faces/{stream_id}")