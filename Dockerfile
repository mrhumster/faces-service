# ============================================================
# builder: компилирует insightface C-extension и собирает venv.
# Нужен build-essential, поэтому изолирован в отдельную стадию.
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# insightface ставим с --no-deps + аккуратно выписанный минимум,
# который реально импортируется веткой detection+recognition
# (см. анализ: onnx — для arcface_onnx.onnx.load(), cv2+skimage —
# для предобработки/align, requests+tqdm — storage helper).
# Затем выпиливаем import mask_renderer из insightface/app/__init__.py:
# он единственный тянет albumentations -> face3d -> matplotlib —
# всё это не нужно для детекции/рикогнишена (мы не рендерим маски).
COPY requirements.txt .
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --no-deps insightface==0.7.3 \
    && .venv/bin/pip install --no-cache-dir \
        numpy==1.26.4 \
        onnx \
        onnxruntime==1.20.1 \
        opencv-python-headless \
        scipy \
        scikit-image \
        requests \
        tqdm \
    && sed -i "/from .mask_renderer import \*/d" /app/.venv/lib/python3.12/site-packages/insightface/app/__init__.py \
    && .venv/bin/pip install --no-cache-dir \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.32.1" \
        pydantic==2.10.4 \
        psycopg2-binary==2.9.10 \
        minio==7.2.12 \
        PyJWT==2.10.1 \
        httpx==0.28.1 \
        prometheus-client==0.21.1 \
    && .venv/bin/pip freeze > /app/installed.txt

# ============================================================
# runtime: худой слой, только venv + модели + код.
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_ROOT=/models \
    PATH="/app/.venv/bin:$PATH"

# libgl1 не нужен (opencv-python-headless); libglib2.0-0 — для cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgthread-2.0-0 \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Bake InsightFace buffalo_l into the image so the pod never hits the internet.
# insightface's ensure_available('models', name, root) resolves the pack to
# $(MODELS_ROOT)/models/<name>, i.e. /models/models/buffalo_l. The release zip
# holds the model files at its top level, so they must be unzipped into that
# exact subdir or insightface falls back to downloading at runtime.
# Only det_10g (detection) + w600k_r50 (recognition) are needed; the landmark
# and genderage ONNX files are ignored by FaceAnalysis(allowed_modules=[...]).
# --mount=type=bind keeps the 276MB zip out of the layer graph entirely.
RUN --mount=type=bind,source=models/buffalo_l.zip,target=/tmp/buffalo_l.zip \
    mkdir -p /models/models/buffalo_l \
    && unzip -o /tmp/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d /models/models/buffalo_l \
    && test -f /models/models/buffalo_l/det_10g.onnx && test -f /models/models/buffalo_l/w600k_r50.onnx \
    && echo "model files present"

COPY main.py requirements.txt ./
COPY app/ ./app/

# Fail fast: the model must actually load during the build.
RUN .venv/bin/python - <<'PY'
import logging
logging.basicConfig(level=logging.WARNING)
import os
os.environ.setdefault("FACES_INTERNAL_TOKEN", "")
from app.model import FaceModel
m = FaceModel()
print("model loaded ok:", sorted(os.listdir("/models/models/buffalo_l")))
PY

EXPOSE 8080

CMD ["python", "main.py"]