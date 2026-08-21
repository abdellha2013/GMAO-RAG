# ═══════════════════════════════════════════════════════════════════
# GMAO-RAG — Dockerfile (multi-stage, uv)
# ═══════════════════════════════════════════════════════════════════
# Build :  docker build -t gmao-rag .
# Run   :  docker run -p 8000:8000 --env-file .env gmao-rag
# ═══════════════════════════════════════════════════════════════════

# ── Étape 1 : installer les dépendances dans une image uv ───────
FROM python:3.13-slim AS builder

# uv pour une installation rapide des dépendances
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copier d'abord pyproject.toml pour profiter du cache Docker
COPY pyproject.toml .

# Installer les dépendances dans un venv hors-ligne
RUN uv venv /app/.venv --python 3.13 && \
    uv pip install --python /app/.venv/bin/python --no-cache -r pyproject.toml

# ── Étape 2 : image finale minimale ─────────────────────────────
FROM python:3.13-slim AS runtime

# Variables d'environnement pour le venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copier le venv depuis l'étape builder
COPY --from=builder /app/.venv /app/.venv

# Copier le code source (pas de tests dans l'image de production)
COPY app/ /app/app/
COPY db/  /app/db/

# Copier l'entrypoint
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Port de l'API
EXPOSE 8000

# Health check intégré à Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
