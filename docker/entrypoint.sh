#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# GMAO-RAG — entrypoint.sh
# ═══════════════════════════════════════════════════════════════════
# Séquence de démarrage :
#   1. Attendre que MySQL soit prêt
#   2. Attendre que Qdrant soit prêt
#   3. Créer la collection Qdrant (si elle n'existe pas)
#   4. Lancer uvicorn
# ═══════════════════════════════════════════════════════════════════

set -e

# ── Couleurs pour les logs ──────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[entrypoint]${NC} $1"; }
warn() { echo -e "${YELLOW}[entrypoint]${NC} $1"; }
err() { echo -e "${RED}[entrypoint]${NC} $1" >&2; }

# ── Variables (avec defaults) ──────────────────────────────────
DB_HOST="${GMAO_DB_HOST:-mysql}"
DB_PORT="${GMAO_DB_PORT:-3306}"
DB_USER="${GMAO_DB_USER:-root}"
DB_PASS="${GMAO_DB_PASSWORD:-gmao_rag_2026}"
DB_NAME="${GMAO_DB_NAME:-gmao_rag}"

QDRANT_HOST="${QDRANT_HOST:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION_NAME:-gmao_chunks}"

EMBEDDING_MODEL="${EMBEDDING_SMALL_MODEL_NAME:-intfloat/multilingual-e5-small}"
EMBEDDING_DIM="${EMBEDDING_SMALL_MODEL_DIMENSION:-384}"

# ═════════════════════════════════════════════════════════════════
# 1. Attendre MySQL
# ═════════════════════════════════════════════════════════════════
log "Attente de MySQL sur ${DB_HOST}:${DB_PORT}..."

MAX_RETRIES=60
RETRY=0
while ! python -c "
import pymysql
pymysql.connect(host='${DB_HOST}', port=${DB_PORT}, user='${DB_USER}', password='${DB_PASS}', database='${DB_NAME}')
" 2>/dev/null; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        err "MySQL n'est pas prêt après ${MAX_RETRIES} tentatives."
        exit 1
    fi
    warn "MySQL pas encore prêt... (${RETRY}/${MAX_RETRIES})"
    sleep 2
done
log "MySQL est prêt."

# ═════════════════════════════════════════════════════════════════
# 2. Attendre Qdrant
# ═════════════════════════════════════════════════════════════════
log "Attente de Qdrant sur ${QDRANT_HOST}:${QDRANT_PORT}..."

RETRY=0
while ! python -c "
import urllib.request
urllib.request.urlopen('http://${QDRANT_HOST}:${QDRANT_PORT}/health')
" 2>/dev/null; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        err "Qdrant n'est pas prêt après ${MAX_RETRIES} tentatives."
        exit 1
    fi
    warn "Qdrant pas encore prêt... (${RETRY}/${MAX_RETRIES})"
    sleep 2
done
log "Qdrant est prêt."

# ═════════════════════════════════════════════════════════════════
# 3. Créer la collection Qdrant (si elle n'existe pas)
# ═════════════════════════════════════════════════════════════════
log "Vérification de la collection Qdrant '${QDRANT_COLLECTION}'..."

python -c "
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

client = QdrantClient(host='${QDRANT_HOST}', port=${QDRANT_PORT})

if client.collection_exists('${QDRANT_COLLECTION}'):
    print('Collection existante, vérification de la compatibilité...')
    col = client.get_collection('${QDRANT_COLLECTION}')
    existing = col.config.params.vectors.size if hasattr(col.config.params.vectors, 'size') else None
    if existing != ${EMBEDDING_DIM}:
        print(f'ATTENTION: dimension existante ({existing}) != attendue (${EMBEDDING_DIM})')
    else:
        print(f'Collection compatible (dimension={existing})')
else:
    print(f'Création de la collection ${QDRANT_COLLECTION} (dimension=${EMBEDDING_DIM})...')
    client.create_collection(
        collection_name='${QDRANT_COLLECTION}',
        vectors_config=VectorParams(size=${EMBEDDING_DIM}, distance=Distance.COSINE),
    )
    # Index de payload pour le filtrage
    client.create_payload_index('${QDRANT_COLLECTION}', 'type_source', PayloadSchemaType.KEYWORD)
    client.create_payload_index('${QDRANT_COLLECTION}', 'id_equipement', PayloadSchemaType.INTEGER)
    print('Collection et index créés avec succès.')
" 2>&1

log "Collection Qdrant prête."

# ═════════════════════════════════════════════════════════════════
# 4. Lancer l'application
# ═════════════════════════════════════════════════════════════════
log "Démarrage de GMAO-RAG API..."
exec uvicorn app.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info
