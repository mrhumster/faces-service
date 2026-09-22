FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_ROOT=/models \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# Bake InsightFace buffalo_l into the image so the pod never hits the internet.
RUN curl -fsSL -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    && mkdir -p /models \
    && unzip -o /tmp/buffalo_l.zip -d /models \
    && rm /tmp/buffalo_l.zip

COPY . .

# Fail fast: the model must actually load during the build.
RUN .venv/bin/python - <<'PY'
import logging
logging.basicConfig(level=logging.WARNING)
import os
os.environ.setdefault("FACES_INTERNAL_TOKEN", "")
from app.model import FaceModel
m = FaceModel()
print("model loaded ok:", sorted(os.listdir("/models/buffalo_l")))
PY

EXPOSE 8080

CMD ["python", "main.py"]