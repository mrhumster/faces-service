import threading
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import psycopg2.pool

from . import config

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_lock = threading.Lock()


def init_db() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            return
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=8,
            host=config.Config.db_host,
            port=config.Config.db_port,
            user=config.Config.db_user,
            password=config.Config.db_pass,
            dbname=config.Config.db_name,
        )


def close_db() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


@contextmanager
def conn():
    init_db()
    if _pool is None:
        raise RuntimeError("db pool not initialized")
    c = _pool.getconn()
    try:
        with c.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            yield cur
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        _pool.putconn(c)


def ping() -> None:
    with conn() as cur:
        cur.execute("SELECT 1")


def _tolist_array(a: list[float]) -> str:
    return "{" + ",".join(f"{v:f}" for v in a) + "}"


def _from_list_array(s) -> list[float]:
    if s is None:
        return []
    if isinstance(s, list):
        return [float(v) for v in s]
    body = s.strip("{}")
    if not body:
        return []
    return [float(v) for v in body.split(",")]


def get_clusters_for_owner(owner_id: str) -> list[dict]:
    with conn() as cur:
        cur.execute(
            """
            SELECT id, name, is_named, centroid, sample_count, crop_object, created_at, updated_at
            FROM clusters
            WHERE owner_id = %s
            ORDER BY is_named DESC, created_at ASC
            """,
            (owner_id,),
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": str(r["id"]),
                "name": r["name"],
                "is_named": r["is_named"],
                "centroid": _from_list_array(r["centroid"]),
                "sample_count": r["sample_count"],
                "crop_object": r["crop_object"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        )
    return out


def get_cluster(cluster_id: str, owner_id: str | None = None) -> dict | None:
    with conn() as cur:
        cur.execute(
            """
            SELECT id, owner_id, name, is_named, centroid, sample_count, crop_object, created_at, updated_at
            FROM clusters
            WHERE id = %s::uuid
            """,
            (cluster_id,),
        )
        r = cur.fetchone()
    if r is None:
        return None
    if owner_id is not None and str(r["owner_id"]) != owner_id:
        return None
    return {
        "id": str(r["id"]),
        "owner_id": str(r["owner_id"]),
        "name": r["name"],
        "is_named": r["is_named"],
        "centroid": _from_list_array(r["centroid"]),
        "sample_count": r["sample_count"],
        "crop_object": r["crop_object"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def get_cluster_total_counts(owner_id: str) -> int:
    with conn() as cur:
        cur.execute("SELECT count(*) FROM clusters WHERE owner_id = %s", (owner_id,))
        return int(cur.fetchone()[0])


def get_cluster_videos(cluster_id: str) -> list[dict]:
    with conn() as cur:
        cur.execute(
            """
            SELECT stream_id, count(*) AS n
            FROM face_occurrences
            WHERE cluster_id = %s::uuid
            GROUP BY stream_id
            ORDER BY n DESC
            """,
            (cluster_id,),
        )
        return [
            {"stream_id": str(r["stream_id"]), "count": int(r["n"])} for r in cur.fetchall()
        ]


def get_cluster_occurrences(cluster_id: str) -> list[dict]:
    with conn() as cur:
        cur.execute(
            """
            SELECT stream_id, t_seconds, confidence, created_at
            FROM face_occurrences
            WHERE cluster_id = %s::uuid
            ORDER BY created_at DESC
            """,
            (cluster_id,),
        )
        return [
            {
                "stream_id": str(r["stream_id"]),
                "t_seconds": float(r["t_seconds"]),
                "confidence": float(r["confidence"]),
                "created_at": r["created_at"],
            }
            for r in cur.fetchall()
        ]


def get_occurrences_for_stream(stream_id: str) -> list[dict]:
    with conn() as cur:
        cur.execute(
            """
            SELECT cluster_id, t_seconds, confidence
            FROM face_occurrences
            WHERE stream_id = %s::uuid
            ORDER BY cluster_id, t_seconds
            """,
            (stream_id,),
        )
        return [
            {
                "cluster_id": str(r["cluster_id"]),
                "t_seconds": float(r["t_seconds"]),
                "confidence": float(r["confidence"]),
            }
            for r in cur.fetchall()
        ]


def rename_cluster(
    cluster_id: str, owner_id: str, name: str | None
) -> tuple[str, str] | None:
    """Rename a cluster owned by owner_id. Returns None on success, or
    (code, message) on conflict (name already owned by same owner)."""
    with conn() as cur:
        cur.execute(
            "SELECT owner_id, name FROM clusters WHERE id = %s::uuid",
            (cluster_id,),
        )
        row = cur.fetchone()
        if row is None or str(row["owner_id"]) != owner_id:
            return ("not_found", "cluster not found")

        if name is not None:
            name = name.strip()
        if name == "":
            name = None

        if name is not None:
            cur.execute(
                """
                SELECT id FROM clusters
                WHERE owner_id = %s AND lower(name) = lower(%s) AND id != %s::uuid
                """,
                (owner_id, name, cluster_id),
            )
            if cur.fetchone() is not None:
                return ("conflict", "name already in use")

        cur.execute(
            "UPDATE clusters SET name = %s, is_named = %s, updated_at = now() WHERE id = %s::uuid",
            (name, name is not None, cluster_id),
        )
        return None


def upsert_cluster(owner_id: str, name: str | None, is_named: bool, centroid: list[float]) -> str:
    with conn() as cur:
        cur.execute(
            """
            INSERT INTO clusters (owner_id, name, is_named, centroid, sample_count, created_at, updated_at)
            VALUES (%s, %s, %s, %s::float8[], 0, now(), now())
            RETURNING id
            """,
            (owner_id, name, is_named, _tolist_array(centroid)),
        )
        cluster_id = cur.fetchone()[0]
        return str(cluster_id)


def append_stream_occurrences(
    owner_id: str,
    stream_id: str,
    items: list[tuple[str, list[float], float, float]],
) -> int:
    """items = (cluster_id, embedding, t_seconds, confidence). Returns count written
    (ON CONFLICT DO NOTHING dedups by (stream_id, cluster_id, t_seconds))."""
    written = 0
    with conn() as cur:
        for cluster_id, embedding, t_seconds, confidence in items:
            cur.execute(
                """
                INSERT INTO face_occurrences
                    (owner_id, stream_id, cluster_id, embedding, t_seconds, confidence, created_at)
                VALUES (%s, %s::uuid, %s::uuid, %s::float8[], %s, %s, now())
                ON CONFLICT (stream_id, cluster_id, t_seconds) DO NOTHING
                """,
                (
                    owner_id,
                    stream_id,
                    cluster_id,
                    _tolist_array(embedding),
                    t_seconds,
                    confidence,
                ),
            )
            written += cur.rowcount or 0
        # recompute sample_count per touched cluster in one statement
        cur.execute(
            """
            UPDATE clusters c SET sample_count = (
                SELECT count(*) FROM face_occurrences o WHERE o.cluster_id = c.id
            ), updated_at = now()
            WHERE c.owner_id = %s AND c.id IN (
                SELECT cluster_id FROM face_occurrences WHERE stream_id = %s::uuid
            )
            """,
            (owner_id, stream_id),
        )
    return written


def get_stream_owner(stream_id: str) -> str | None:
    """Looks up the stream owner from the faces DB index if present."""
    with conn() as cur:
        cur.execute(
            "SELECT DISTINCT owner_id FROM face_occurrences WHERE stream_id = %s::uuid LIMIT 1",
            (stream_id,),
        )
        r = cur.fetchone()
    return str(r[0]) if r else None


def update_cluster_centroid(cluster_id: str, owner_id: str, centroid: list[float]) -> None:
    with conn() as cur:
        cur.execute(
            "UPDATE clusters SET centroid = %s::float8[], updated_at = now() WHERE id = %s::uuid AND owner_id = %s",
            (_tolist_array(centroid), cluster_id, owner_id),
        )


def set_cluster_crop(cluster_id: str, owner_id: str, crop_object: str | None) -> None:
    """Point a cluster to its face crop object in MinIO (or clear it)."""
    with conn() as cur:
        cur.execute(
            "UPDATE clusters SET crop_object = %s, updated_at = now() WHERE id = %s::uuid AND owner_id = %s",
            (crop_object, cluster_id, owner_id),
        )


def cascade_stream(stream_id: str) -> dict:
    """Permanently delete all faces data for a stream: occurrences plus any
    clusters orphaned by the removal (sample_count dropping to 0). Returns
    counts of deleted occurrences/clusters plus the removed clusters
    ({owner_id, id}) so the caller can also purge their crop images."""
    deleted_clusters: list[dict] = []
    with conn() as cur:
        cur.execute(
            "SELECT DISTINCT cluster_id FROM face_occurrences WHERE stream_id = %s::uuid",
            (stream_id,),
        )
        cluster_ids = [r[0] for r in cur.fetchall()]

        cur.execute(
            "DELETE FROM face_occurrences WHERE stream_id = %s::uuid",
            (stream_id,),
        )
        occurrences_deleted = cur.rowcount or 0

        clusters_deleted = 0
        for cid in cluster_ids:
            if cid is None:
                continue
            cur.execute(
                "SELECT owner_id FROM clusters WHERE id = %s::uuid",
                (cid,),
            )
            row = cur.fetchone()
            cur.execute(
                """
                UPDATE clusters SET sample_count = (
                    SELECT count(*) FROM face_occurrences WHERE cluster_id = %s::uuid
                ), updated_at = now()
                WHERE id = %s::uuid
                """,
                (cid, cid),
            )
            cur.execute(
                "DELETE FROM clusters WHERE id = %s::uuid AND sample_count = 0",
                (cid,),
            )
            if cur.rowcount:
                clusters_deleted += cur.rowcount
                if row is not None:
                    deleted_clusters.append(
                        {"owner_id": str(row["owner_id"]), "id": str(cid)}
                    )

    return {
        "occurrences_deleted": occurrences_deleted,
        "clusters_deleted": clusters_deleted,
        "deleted_clusters": deleted_clusters,
    }