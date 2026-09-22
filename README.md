# faces-service

Python face-recognition service for GoCast ("Свои люди" — find videos by face).

## What it does

- **Reader** (FastAPI, port 8080): REST API for a video owner to list their detected
  people clusters, drill into a cluster (which videos / timestamps a person appears at),
  rename a cluster (merge is just rename onto the same person).
- **`POST /infer`** (internal, gated by `X-Internal-Token`): the Go `faces-worker` pushes
  sampled frames (already uploaded to MinIO under a prefix), this service runs
  **InsightFace buffalo_l** (detect + 512-d embedding) per frame, clusters embeddings with
  numpy cosine similarity (threshold: 0.4 vs named clusters, 0.5 vs anonymous ones),
  and persists `clusters` + `face_occurrences` in the `faces` Postgres database.

## Architecture

Frames never hit Go — `faces-worker` uploads them to MinIO and calls the internal
`/infer` endpoint. This service is CPU-only (onnxruntime CPUExecutionProvider), runs in
the k3s `go-app` namespace as `faces-reader`.

Storage:
- Postgres database **`faces`** (own DB, schema via `db-migrate` target `faces`):
  `clusters(owner_id, name, is_named, centroid float8[], sample_count)` and
  `face_occurrences(owner_id, stream_id, cluster_id, embedding float8[], t_seconds, confidence)`.
- MinIO: frames read from the same bucket Go writes them to.

## Auth

Reader endpoints require a Bearer JWT (identity-service public key fetched from
`JWT_ACCESS_PUBLIC_KEY_URL` at startup, same pattern as events-service). Everything is
scoped by `owner_id` from the `user_id` claim; admin (`role == "admin"`) bypasses the
owner check.

`POST /infer` requires the shared internal token in `X-Internal-Token` (env
`FACES_INTERNAL_TOKEN`) — fail-closed: empty token config => 503.

## API

- `GET /health` -> `{"status":"up"}` (with DB ping)
- `GET /metrics` -> Prometheus
- `GET /faces` -> list clusters of the caller: `{clusters:[{id,name,is_named,sample_count,first_seen,last_seen,video_count}], total}`
- `GET /faces/:id` -> cluster detail + occurrences: `{cluster:{...}, occurrences:[{stream_id,title?,t_seconds,confidence,frame_prefix}]}`
- `PATCH /faces/:id` `{name}` -> rename cluster (409 on name already used by the same owner)
- `POST /infer` internal: `{stream_id, owner_id, frames_prefix, count}`

`frame_prefix` lets the frontend render the exact sampled frame via its
`STORAGE_URL/bucket/<frames_prefix>/frame_0000.jpg` (index derived from `t_seconds`).

## Config (env)

See `deploy/k8s/deployment.yaml` for the full set. Highlights:
`DB_*` (db `faces`), `MINIO_*` (endpoint/keys/bucket — same bucket faces-worker uploads to),
`MODELS_ROOT` (default `/models`, buffalo_l baked into image at build),
`JWT_ACCESS_PUBLIC_KEY_URL`, `FACES_INTERNAL_TOKEN`,
`FACES_MATCH_THRESHOLD` (0.4), `FACES_UNKNOWN_THRESHOLD` (0.5).

## Build & deploy

```bash
make build            # docker build (downloads buffalo_l, ~700MB image)
make push             # xomrkob/faces-service:latest (+ git describe tag)
make deploy           # kubectl rollout deploy/faces-reader
```

Image bakes InsightFace buffalo_l at build time so the pod never hits the internet.