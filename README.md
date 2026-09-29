# faces-service

Python face-recognition service for GoCast: it turns sampled video frames into
**people** — clusters of 512-d face embeddings — and exposes them as a
curatable REST API ("who appears in this video", rename, merge, detach, delete).

CPU-only: InsightFace **buffalo_l** (detection + recognition) on
`onnxruntime` `CPUExecutionProvider`. Frames never pass through Go — the
`faces-worker` uploads them to MinIO and calls the internal `/infer` endpoint,
which reads the frames back, runs the model, and persists results in Postgres.

One FastAPI app on port `8080` (`UVICORN_WORKERS=4` locally; the Kubernetes
deployment pins `UVICORN_WORKERS=1`), deployment `faces-reader` in the k3s
`go-app` namespace. No queue of its own: the worker is the producer.

## Architecture

```
faces-service/
  app/
    __init__.py       FastAPI app: lifespan, CORS, AuthError + DB-busy handlers, routers
    config.py         env-backed Config (class attributes read at import)
    auth.py           JWT verification (identity public key), ensure_owner, internal token
    db.py             psycopg2 pool + cluster/occurrence queries (semaphore-guarded)
    model.py          FaceModel: InsightFace buffalo_l app (det + rec, CPU, det_size 640)
    infer.py          InferEngine.run(): frame loop, matching, persistence, counters
    clustering.py     cosine similarity, union-find groups, centroid merge
    minio_client.py   FrameStore: read frames, get/delete crops, delete stream frames
    metrics.py        Prometheus collectors
    routes/
      health.py       GET /health
      metrics.py      GET /metrics
      reader.py       public owner/admin API (/faces, /streams/{id}/faces, ...)
      infer.py        internal API (POST /infer, cascade delete)
  main.py             uvicorn entry point
  deploy/k8s/         deployment.yaml, service.yaml
  requirements.txt    pinned runtime deps
Dockerfile, Makefile (xomrkob/faces-service)
```

Schema migrations live in **db-migrate** (target `faces`, version table
`schema_migrations_faces`, dedicated Postgres database `faces`).

## REST API

Served on `SERVER_ADDR` (default `:8080`). No global path prefix. Errors use
FastAPI's `{"detail": "..."}` shape.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | none | liveness + Postgres ping |
| `GET` | `/metrics` | none | Prometheus exposition |
| `GET` | `/faces` | Bearer | paginated people list + similarity groups |
| `POST` | `/faces/delete-empty` | Bearer | delete clusters with no videos |
| `GET` | `/faces/suggest` | Bearer | named-cluster name autocomplete |
| `GET` | `/faces/{cluster_id}` | Bearer (owner/admin) | cluster + occurrences + videos |
| `PATCH` | `/faces/{cluster_id}` | Bearer (owner/admin) | rename (or clear name) |
| `GET` | `/faces/{cluster_id}/crop` | Bearer (owner/admin) | face-crop image |
| `PUT` | `/faces/{cluster_id}/crop` | Bearer (owner/admin) | replace face-crop image |
| `GET` | `/streams/{stream_id}/faces` | Bearer (owner/admin) | people in one video |
| `DELETE` | `/streams/{stream_id}/faces/{cluster_id}` | Bearer (owner/admin) | detach person from one video |
| `DELETE` | `/faces/{cluster_id}` | Bearer (owner/admin) | delete a person |
| `POST` | `/faces/merge` | Bearer (owner/admin) | merge several people into one |
| `POST` | `/infer` | `X-Internal-Token` | run detection over sampled frames |
| `POST` | `/streams/{stream_id}/faces/delete` | `X-Internal-Token` | cascade delete on stream removal |

### People list — `GET /faces`

Query: `limit` (default `50`, `1..100`), `offset` (default `0`, `>= 0`).
Returns clusters that appear in **at least one video**; zero-video "orphan"
clusters are hidden and reported separately.

```json
{
  "clusters": [ { /* cluster + video_count + videos, centroid stripped */ } ],
  "groups":   [ { "rep": {...}, "members": [{"cluster": {...}, "sim": 0.93}], "maxSim": 0.97 } ],
  "total": 42,          // video-bearing clusters
  "empty_count": 3,     // orphans, hidden
  "rest_total": 30,     // non-grouped clusters (pagination bound)
  "limit": 50,
  "offset": 0
}
```

- Flat `clusters` are sorted: named first, then by name, then by descending
  `sample_count`.
- Similarity **groups** (near-duplicate people) are computed server-side and
  returned in full on every page; grouped clusters are excluded from the flat
  list (see [Clustering](#clustering--matching)).
- Centroid vectors are stripped from list responses to keep payloads small.

### Cluster detail — `GET /faces/{cluster_id}`

`200 {cluster, occurrences, videos}`. A cluster row is
`{id, owner_id, name, is_named, centroid, sample_count, crop_object,
created_at, updated_at}` (the list endpoints strip `centroid`).

### Rename — `PATCH /faces/{cluster_id}`

Body `{"name": "Artem Khomyakov"}` (`name` may be `null` to clear it,
`max_length=120`). The name is trimmed; empty → unnamed. `409` if the owner
already has a cluster with that name (case-insensitive). Returns the updated
cluster.

### Face crop

- `GET /faces/{cluster_id}/crop` → image bytes, `Cache-Control: private,
  max-age=3600`. `404 no crop yet` if none stored; `503 storage unavailable`
  if the MinIO store isn't initialized.
- `PUT /faces/{cluster_id}/crop` → multipart `file` (raw image bytes, max
  `5 MiB`). Replaces the stored crop. `413 file too large`, `400 invalid
  image` if it doesn't decode.

### People in a video — `GET /streams/{stream_id}/faces`

`200 {clusters: [{cluster, count, occurrences}, ...]}` for the caller. Rows for
foreign clusters are skipped silently (defense in depth alongside owner auth).

### Detach from one video — `DELETE /streams/{stream_id}/faces/{cluster_id}`

Removes the cluster's occurrences **in that stream only**. If the cluster has no
occurrences left in any stream it is deleted entirely along with its crop.

```json
{ "cluster_id": "...", "stream_id": "...",
  "removed": 12, "sample_count": 30, "cluster_deleted": false }
```

The stream's `faces_detected` flag is intentionally **left unchanged** — this is
curation, not a signal to re-run detection.

### Delete a person — `DELETE /faces/{cluster_id}`

Deletes the cluster, all its occurrences, and its crop. Returns
`{cluster_id}`. `faces_detected` is again left unchanged (detection already ran).

### Merge — `POST /faces/merge`

Body `{"cluster_ids": [...]}` (2–100 ids, all owned by the caller). Occurrences
move to the **target** = the first selected cluster that has a name, else the
first id. Source clusters are deleted with their crops. Returns
`{cluster, merged_ids, target_id}`.

### Autocomplete — `GET /faces/suggest`

Query `q` (`min_length=3`, `max_length=120`), optional `exclude` (cluster id).
Returns `{clusters: [{id, name, is_named, sample_count, crop_object}]}` — the
caller's **named** clusters whose name matches `q` as a prefix, case-
insensitively (escaped `LIKE`), ordered by name. Used by the UI to warn about
renaming to an existing person and offer "merge into".

### Delete empty clusters — `POST /faces/delete-empty`

Deletes the caller's clusters that have no video occurrences (orphans left over
from detection). Returns `{deleted: N}`; idempotent (repeat → `0`).

### Internal: inference — `POST /infer`

Requires header `X-Internal-Token` equal to `FACES_INTERNAL_TOKEN`. Unset config
→ `503 internal token not configured`; mismatch → `401 invalid internal token`.
Body:

```json
{ "stream_id": "...", "owner_id": "...",
  "frames_prefix": "faces/<stream_id>", "count": 42 }
```

`count` must satisfy `0 < count <= FACES_MAX_FRAMES` (else `400 count out of
range`). If the model engine isn't ready → `503 engine not ready`. Response:

```json
{ "faces_detected": 9, "frames_processed": 42, "frames_skipped": 0,
  "occurrences_written": 9, "clusters_created": ["<new-id>"], "crops_saved": 3 }
```

`clusters_created` is a list of newly created cluster ids.

### Internal: cascade delete — `POST /streams/{stream_id}/faces/delete`

Called by `faces-worker` after a stream is deleted. Drops all face data for the
stream, deletes its stored frames (`faces/<stream_id>` prefix) and the crops of
any clusters that lose their last occurrence. Returns
`{stream_id, occurrences_deleted, clusters_deleted, deleted_clusters}`.

## Auth

Two independent schemes:

- **Reader endpoints** — Bearer JWT verified with the identity-service RSA
  public key, fetched **once at startup** (10s timeout) from
  `JWT_ACCESS_PUBLIC_KEY_URL`. Tokens are only accepted via the
  `Authorization: Bearer` header (no `?token=` fallback). Data is scoped by the
  `user_id` claim; `role == "admin"` bypasses owner checks.
  - Missing/invalid/expired token → `401`; a token without a `user_id` claim →
    `401 token missing user_id`.
  - `ensure_owner` returns **`404`** for a foreign cluster (not `403`) so the
    API never discloses whether a cluster exists for someone else.
  - If the key could not be fetched at startup (URL unset or non-200), the
    failure is logged and **deferred**: the pod still starts, but every reader
    call answers `503 token service unavailable` until a restart succeeds.
- **Internal endpoints** (`/infer`, cascade delete) — `X-Internal-Token` header
  compared to `FACES_INTERNAL_TOKEN`; missing/mismatched token → `401`; the
  variable being **unset** → `503 internal token not configured` (fail-closed:
  the endpoint is never open when no secret is configured).

### CORS

`CORS_ALLOW_ORIGINS` is a comma-separated allowlist; an empty value (the code
default) resolves to `*`, and `allow_credentials` stays enabled. The k3s
deployment does not rely on the code default: the gocast-infra root script
`scripts/render-env.sh` renders an explicit
`http://localhost:5173,https://example.com,https://api.example.com` list into
`faces-service-config`.

## Clustering & matching

`app/clustering.py` (pure numpy, unit-tested):

- **Similarity groups** for the people list: union-find over pairs whose
  centroid cosine similarity is `>= 0.5`. Each group's representative is the
  named cluster with the largest `sample_count` (else the max-sample one);
  members carry their `sim` to the rep, and `maxSim` orders the groups.
- **Matching a detection** during inference: an embedding is compared by
  cosine similarity to the owner's existing cluster centroids —
  - against **named** clusters: `>= FACES_MATCH_THRESHOLD` (default `0.4`);
  - against **anonymous** clusters: `>= FACES_UNKNOWN_THRESHOLD` (default `0.5`).
  The best match wins; otherwise a new cluster is created and its crop saved.
- **Centroid drift during inference**: a matched cluster's centroid is updated as
  a running (unnormalized) mean — `(old*sample_count + new) / (sample_count+1)` —
  and persisted only if a new sample was added. The manual
  `POST /faces/merge` does **not** touch centroids: the target keeps its own
  centroid and only `sample_count` is recomputed.

## Inference pipeline

`InferEngine.run()` (per frame):

1. Read `faces/<stream_id>/frame_%05d.jpg` from MinIO (`None` → skipped).
2. Run InsightFace `FaceModel` detection (`det_size=(640,640)`,
   `FACES_DETECT_THRESHOLD`) → up to N face boxes per frame.
3. Crop each face, compute the 512-d embedding.
4. Match against the owner's centroids (see above); assign or create a cluster.
5. Persist an occurrence `(stream_id, cluster_id, t_seconds, confidence,
   embedding)`; de-duplicated on `(stream_id, cluster_id, t_seconds)`.
6. Save a face crop `faces/crops/{owner_id}/{cluster_id}.jpg` when the cluster
   was just created, or when a matched cluster still has `crop_object IS NULL`
   (e.g. its crop was purged) — skipped for detections without a bounding box,
   and a cluster is cropped at most once.

Frame timestamps are `t = (index + 0.5) * SAMPLE_INTERVAL_SECONDS`, with
`SAMPLE_INTERVAL_SECONDS = 5.0` hardcoded in `app/infer.py` — it must stay in
sync with `faces-worker`'s `SampleIntervalSeconds` (also `5.0`, Go constant, no
env var on either side). The worker has **no** frame cap of its own: it uploads
every frame it extracted and sends that total as `count`, which this service
then validates — a `count` outside `1..FACES_MAX_FRAMES` (default 500) is
rejected with `400 count out of range`.

## Database

Dedicated Postgres database **`faces`**. Pool via `psycopg2` with a
`ThreadedConnectionPool`; a `threading.BoundedSemaphore` caps concurrent
in-flight queries and a timeout raises `DBBusyError` → `503 database busy`
(handled globally).

### Schema (db-migrate target `faces`)

- **`clusters`** — one "person" per owner: `id uuid PK`, `owner_id uuid NOT NULL`,
  `name text NULL`, `is_named boolean NOT NULL DEFAULT false`,
  `centroid float8[] NOT NULL`, `sample_count integer NOT NULL DEFAULT 0`,
  `crop_object text` (added in `0002_cluster_crop`), `created_at`/`updated_at`.
  Indexed by `owner_id`, by `(owner_id) WHERE name IS NOT NULL`, and unique
  case-insensitively on `(owner_id, lower(name)) WHERE name IS NOT NULL`.
- **`face_occurrences`** — one row per face sighting: `id uuid PK`,
  `owner_id uuid NOT NULL`, `stream_id uuid NOT NULL`, `cluster_id uuid` (FK →
  `clusters`, `ON DELETE CASCADE`; the column is nullable, though in practice
  always set), `embedding float8[]`, `t_seconds double precision`,
  `confidence double precision`, `created_at`. Indexed by owner and by
  `(stream_id, created_at DESC)`; unique on `(stream_id, cluster_id,
  t_seconds)`, which is what makes inference re-runs idempotent.

`append_stream_occurrences` writes in a transaction and recomputes
`sample_count` for every touched cluster. Deleting a cluster cascades its
occurrences; detach/merge move them explicitly.

## Object storage (MinIO)

One shared bucket with the Go `faces-worker` that produces the frames
(`MINIO_BUCKET_NAME`; code default `stream-service-test`, the k3s deployment
renders `go-app-bucket`):

| Object | Key | Used by |
| --- | --- | --- |
| sampled frame | `faces/<stream_id>/frame_%05d.jpg` | inference reads |
| face crop | `faces/crops/<owner_id>/<cluster_id>.jpg` | crop GET/PUT, delete/merge/detach cleanup |

Read failures for a frame return `None` (frame skipped, not fatal). Object
deletion ignores `NoSuchKey`/`NotFound`.

## Configuration (env)

| Var | Default | Description |
| --- | --- | --- |
| `SERVER_ADDR` | `:8080` | HTTP listen address |
| `MODE` | `release` | app mode label |
| `DB_HOST` / `DB_PORT` | `localhost` / `5432` | Postgres |
| `DB_USER` / `DB_PASS` | `postgres` / `` | from secret `go-app-secret` |
| `DB_NAME` | `faces` | dedicated database |
| `DB_MAX_CONN` | `40` | pool size |
| `DB_ACQUIRE_TIMEOUT` | `5` | seconds to wait for a pooled connection |
| `READER_THREADS` | `80` | AnyIO thread-limiter size (FastAPI sync handlers) |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `admin` / `minio123` | MinIO creds |
| `MINIO_BUCKET_NAME` | `stream-service-test` | shared frames/crops bucket (k3s: `go-app-bucket`) |
| `MINIO_USE_SSL` | `false` | MinIO over TLS |
| `JWT_ACCESS_PUBLIC_KEY_URL` | `` | identity public key; unset/non-200 ⇒ reader API answers `503` |
| `CORS_ALLOW_ORIGINS` | `` | comma-separated allowlist (empty → `*`) |
| `FACES_INTERNAL_TOKEN` | `` | shared secret for `/infer` + cascade; unset ⇒ `503` |
| `STREAM_SERVICE_URL` | `http://stream-service:80` | declared but unused — no outbound calls to stream-service |
| `MODELS_ROOT` | `/models` | buffalo_l lives in `$(MODELS_ROOT)/models/buffalo_l` |
| `FACES_MATCH_THRESHOLD` | `0.4` | cosine match vs named clusters |
| `FACES_UNKNOWN_THRESHOLD` | `0.5` | cosine match vs anonymous clusters |
| `FACES_MAX_FRAMES` | `500` | max frames per `/infer` request |
| `FACES_DETECT_THRESHOLD` | `0.4` | InsightFace detection confidence threshold |

Numeric env values are parsed at import; a malformed value is fatal with a
ValueError. `SSLMODE=disable` and `timezone=UTC` are hard-coded in the DSN.

## Metrics

`GET /metrics` on the main HTTP server (Prometheus text exposition, no separate
port). Collectors (labels in parentheses):

- `faces_infer_total{status}` — `/infer` requests by outcome.
- `faces_detected_total` — faces detected across inferences.
- `faces_clusters_created_total` — new clusters.
- `faces_occurrences_written_total` — occurrences persisted.
- `faces_infer_duration_seconds` — inference latency histogram.
- `faces_reader_requests_total{endpoint,status}` — reader REST requests.
- `faces_cascade_total{status}` — internal cascade-delete requests.

The `Counter`s (without labels) are registered at import; the labeled `CounterVec`
series appear only after their first use, so an idle deployment legitimately
exposes few series.

## Health

`GET /health` calls `db.ping_direct()` (fresh connection, bypassing the pool)
and returns `200 {"status":"up"}` or `200 {"status":"down"}` on failure (no
HTTP error). Used by both the liveness and readiness probes.

## Errors

| Situation | Response |
| --- | --- |
| missing/invalid Bearer or internal token | `401 {"detail": ...}` |
| `FACES_INTERNAL_TOKEN` unset | `503 internal token not configured` |
| reader token service not initialized (key fetch failed at startup) | `503 token service unavailable` |
| foreign or unknown cluster (ensure_owner) | `404 not found` |
| rename name already in use (case-insensitive) | `409 {"detail": ...}` |
| `/infer` `count` outside `1..FACES_MAX_FRAMES` | `400 count out of range` |
| crop > 5 MiB / undecodable image | `413` / `400` |
| DB pool exhausted | `503 database busy` (global handler) |
| MinIO store uninitialized | `503 storage unavailable` |
| engine not initialized (infer) | `503 engine not ready` |
| cascade delete failure (DB or storage purge) | `500 cascade delete failed` / `500 cascade storage delete failed` |

`PoolError` (psycopg2) and `DBBusyError` share the `503 database busy` handler
so pool starvation surfaces as a clean 503 rather than a 500.

## Deployment

`deploy/k8s/deployment.yaml` — deployment `faces-reader` in `go-app`, 1 replica,
image `xomrkob/faces-service:latest`, `imagePullPolicy: Always`, container port
`8080`. Prometheus pod annotations scrape `:8080/metrics`.

- Readiness: `httpGet /health:8080`, initial delay 10s, period 10s, timeout 5s,
  failure threshold 3.
- Liveness: same, initial delay 20s, period 15s.
- Requests `200m` / `512Mi`, limits `1` CPU / `2Gi`.

`deploy/k8s/service.yaml` — ClusterIP Service `faces-service`, port `80` →
target port `8080`, selector `app: faces-reader`.

Environment comes from ConfigMap `faces-service-config` (`envFrom`), with
`DB_USER`/`DB_PASS` from secret `go-app-secret`, MinIO creds from
`minio-credentials` and the internal token from `faces-internal-token`. The
ConfigMap, the `faces-service` ingress, and the `db-migrate-faces` Job are
generated in the **gocast-infra** repo (`scripts/render-env.sh`,
`make apply-db-migrate`) — there is no render script inside this service.

## Makefile

```bash
make venv    # python3 -m venv .venv + install -r requirements.txt
make lint    # py_compile app/*.py app/routes/*.py main.py
make build   # docker build -t xomrkob/faces-service:latest .
make push    # docker push
make deploy  # build + push, kubectl apply deploy/k8s/, wait for faces-reader
```

There is no `make test`; the clustering unit tests are run with `pytest`.

## Tests

`tests/test_clustering.py` covers the pure clustering helpers: similarity-group
construction, named-representative preference, member ordering by similarity,
`maxSim` ordering, and the empty / no-centroid edge cases. Run with:

```bash
python3 -m pytest tests/ -q
```

(Docker note: the runtime image has no dev extras beyond the pinned
requirements, and `insightface` is heavy — install `requirements.txt` plus
`pytest` in a venv via `make venv` for local test runs.)

## Related

- **faces-worker** (Go, in `services/faces-worker`): extracts frames
  (`SampleIntervalSeconds = 5.0`), uploads them to MinIO, calls `/infer`,
  reports progress. Its sampling period must match `SAMPLE_INTERVAL_SECONDS` in
  `app/infer.py`; the frame cap (`FACES_MAX_FRAMES`, default 500) is applied by
  this service.
- **stream-service**: owns `streams.faces_detected`; calls the cascade-delete
  endpoint when a stream is removed; triggers the worker.
- **db-migrate** — target `faces` (migrations `0001_clusters_occurrences`,
  `0002_cluster_crop`).
- **web-frontend** — `/people` (list, groups, merge, delete, detach), "People in
  this video" block on the stream page, name-merge suggestion prompt.
