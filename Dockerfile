# Image de SERVICE (inférence + UI). Pas d'entraînement ici : l'entraînement se
# fait sur la machine hôte, sur CPU (cf. docs/decisions/ADR-002), et le modèle
# promu est monté en volume — une image ne doit pas figer un champion.
#
# `tensorflow-macos` n'existe que pour Apple Silicon : l'image Linux installe
# `tensorflow-cpu`, d'où le fichier de dépendances dédié.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COACHVOCAL_ROOT=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY app/ ./app/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps -e .

# Les modèles et les données arrivent par volume (docker-compose.yml)
VOLUME ["/app/artifacts", "/app/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "coachvocal.serving.api:api", "--host", "0.0.0.0", "--port", "8000"]
