#!/usr/bin/env python3
"""Tests manuels de TOUS les endpoints de l'API GMAO-RAG (FastAPI, port 8000).

Ce script est autonome : il ne nécessite que la bibliothèque ``requests``
et un serveur GMAO-RAG démarré localement.

Exécution :
    cd /home/abdellah-daif/GMAO-RAG
    PYTHONPATH=/home/abdellah-daif/GMAO-RAG .venv/bin/python tests/manual/all_api.py

Options :
    --no-color    désactive les couleurs ANSI (utile dans les logs CI)

Sections testées :
    1.  Santé        GET  /api/v1/health              (sans auth)
    2.  Stratégies   GET  /api/v1/strategies          (auth)
    3.  Stats        GET  /api/v1/stats               (auth)
    4.  Documents    GET/DELETE /api/v1/documents/    (auth)
    5.  Retrieve     POST /api/v1/rag/retrieve        (auth)
    6.  Rerank       POST /api/v1/rag/rerank          (auth)
    7.  Search       POST /api/v1/rag/search          (auth)
    8.  Ingest file  POST /api/v1/ingest/file         (auth, multipart)
    9.  Ingest DB    POST /api/v1/ingest/database     (auth)
    10. Ingest files POST /api/v1/ingest/files        (auth)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

# =====================================================================
# Configuration
# =====================================================================

BASE_URL = "http://localhost:8000"
API_KEY = "Zzdv0632"

# En-têtes standard pour les requêtes JSON authentifiées.
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}
JSON_HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Timeouts : la toute première requête peut être lente (réveil du serveur,
# chargement des modèles), et l'ingestion (embedding + Qdrant) ou la
# génération LLM le sont toujours → on laisse 60 s / 120 s.
TIMEOUT_DEFAULT = 60
TIMEOUT_LONG = 120

# Fichiers temporaires utilisés par les tests d'ingestion.
TMP_FILE_1 = Path("/tmp/test_manual_api.txt")
TMP_FILE_2 = Path("/tmp/test_manual_api_2.txt")

# Couleurs ANSI (désactivables via --no-color ou NO_COLOR).
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
if "--no-color" in sys.argv:
    USE_COLOR = False

GREEN = "\033[92m" if USE_COLOR else ""
RED = "\033[91m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
CYAN = "\033[96m" if USE_COLOR else ""
DIM = "\033[2m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""

# Compteurs globaux de résultats.
PASSED = 0
FAILED = 0


# =====================================================================
# Helpers : affichage et assertions
# =====================================================================

def show(label: str, resp: requests.Response, max_body: int = 300) -> None:
    """Affiche le statut HTTP, le temps de traitement serveur et le corps tronqué."""
    process_time = resp.headers.get("X-Process-Time", "?")
    print(f"{DIM}  ↳ {label}{RESET}")
    print(f"{DIM}    Status        : {resp.status_code}{RESET}")
    print(f"{DIM}    X-Process-Time: {process_time} s{RESET}")
    try:
        body = json.dumps(resp.json(), ensure_ascii=False)
    except Exception:
        body = resp.text
    if len(body) > max_body:
        body = body[:max_body] + " …[tronqué]"
    print(f"{DIM}    Body          : {body}{RESET}")


def assert_test(name: str, condition: bool) -> None:
    """Enregistre et affiche le résultat d'un test (PASS/FAIL colorés)."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  {GREEN}✓ PASS{RESET}  {name}")
    else:
        FAILED += 1
        print(f"  {RED}✗ FAIL{RESET}  {name}")


def section(title: str) -> None:
    """Affiche un en-tête de section bien visible."""
    print(f"\n{CYAN}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{RESET}")


def warn(message: str) -> None:
    """Affiche un avertissement (sans faire échouer le test)."""
    print(f"  {YELLOW}⚠ WARNING{RESET}  {message}")


class _FailedResponse:
    """Réponse factice retournée quand le serveur est injoignable.

    Permet au script de continuer et de reporter des FAIL lisibles
    au lieu de planter sur une ConnectionError.
    """

    status_code = 0
    headers: dict = {}
    text = "<serveur injoignable>"

    def json(self) -> dict:
        return {"error": "connexion impossible au serveur"}


def http(method: str, path: str, *, timeout: int = TIMEOUT_DEFAULT,
         **kwargs) -> requests.Response | _FailedResponse:
    """Envoie une requête HTTP en captant les erreurs de connexion."""
    url = f"{BASE_URL}{path}"
    try:
        return requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        print(f"  {RED}ERREUR CONNEXION{RESET} {method} {url} : {exc}")
        return _FailedResponse()


def load_db_config() -> dict:
    """Charge la config MySQL depuis .env (fallback : valeurs locales par défaut).

    Le test d'ingestion base de données doit pointer vers la même base
    que le serveur, on lit donc les variables GMAO_DB_* du fichier .env
    situé à la racine du projet.
    """
    config = {
        "host": "localhost",
        "port": 3306,
        "database": "gmao_rag",
        "user": "root",
        "password": "",
    }
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.split("#")[0].strip()
            if key == "GMAO_DB_HOST" and value:
                config["host"] = value
            elif key == "GMAO_DB_PORT" and value.isdigit():
                config["port"] = int(value)
            elif key == "GMAO_DB_NAME" and value:
                config["database"] = value
            elif key == "GMAO_DB_USER" and value:
                config["user"] = value
            elif key == "GMAO_DB_PASSWORD":
                config["password"] = value
    return config


# =====================================================================
# Section 1 : Health — GET /api/v1/health (sans authentification)
# =====================================================================

def test_health() -> None:
    section("SECTION 1 : HEALTH — GET /api/v1/health")

    resp = http("GET", "/api/v1/health")
    show("health sans auth", resp)

    assert_test("health → 200", resp.status_code == 200)
    if resp.status_code == 200:
        body = resp.json()
        assert_test(
            "health → clés status/qdrant/mysql/version présentes",
            all(k in body for k in ("status", "qdrant", "mysql", "version")),
        )
        warn(f"état global : {body.get('status')} "
             f"(qdrant={'ok' if body.get('qdrant') == 'ok' else 'KO'}, "
             f"mysql={'ok' if body.get('mysql') == 'ok' else 'KO'})")


# =====================================================================
# Section 2 : Strategies — GET /api/v1/strategies (auth requise)
# =====================================================================

def test_strategies() -> None:
    section("SECTION 2 : STRATEGIES — GET /api/v1/strategies")

    # --- Cas nominal ---
    resp = http("GET", "/api/v1/strategies", headers=JSON_HEADERS)
    show("strategies avec auth", resp)
    assert_test("strategies avec auth → 200", resp.status_code == 200)
    if resp.status_code == 200:
        body = resp.json()
        assert_test(
            "strategies → retrieval/reranker/llm/embedding sont des listes",
            all(isinstance(body.get(k), list)
                for k in ("retrieval", "reranker", "llm", "embedding")),
        )

    # --- Sans en-tête Authorization → 422 (champ Header obligatoire) ---
    resp = http("GET", "/api/v1/strategies")
    show("strategies sans auth", resp)
    assert_test("strategies sans auth → 422", resp.status_code == 422)

    # --- Mauvais token → 401 ---
    resp = http("GET", "/api/v1/strategies",
                headers={"Authorization": "Bearer MAUVAIS_TOKEN"})
    show("strategies mauvais token", resp)
    assert_test("strategies mauvais token → 401", resp.status_code == 401)

    # --- En-tête malformé (sans préfixe "Bearer ") → 401 ---
    resp = http("GET", "/api/v1/strategies",
                headers={"Authorization": API_KEY})
    show("strategies header malformé", resp)
    assert_test("strategies header malformé → 401", resp.status_code == 401)


# =====================================================================
# Section 3 : Stats — GET /api/v1/stats (auth requise)
# =====================================================================

def test_stats() -> None:
    section("SECTION 3 : STATS — GET /api/v1/stats")

    resp = http("GET", "/api/v1/stats", headers=JSON_HEADERS)
    show("stats avec auth", resp)

    assert_test("stats avec auth → 200", resp.status_code == 200)
    if resp.status_code == 200:
        body = resp.json()
        assert_test("stats → documents_count >= 0",
                    isinstance(body.get("documents_count"), int)
                    and body["documents_count"] >= 0)
        assert_test("stats → chunks_count >= 0",
                    isinstance(body.get("chunks_count"), int)
                    and body["chunks_count"] >= 0)

    # --- Sans auth → 422 ---
    resp = http("GET", "/api/v1/stats")
    show("stats sans auth", resp)
    assert_test("stats sans auth → 422", resp.status_code == 422)


# =====================================================================
# Section 4 : Documents — GET/DELETE /api/v1/documents/ (auth requise)
# =====================================================================

def test_documents() -> None:
    section("SECTION 4 : DOCUMENTS — GET/DELETE /api/v1/documents/")

    # --- Liste (attention au slash final obligatoire !) ---
    resp = http("GET", "/api/v1/documents/", headers=JSON_HEADERS)
    show("documents liste", resp)
    assert_test("documents liste → 200", resp.status_code == 200)
    if resp.status_code == 200:
        body = resp.json()
        assert_test("documents liste → clés documents/total présentes",
                    "documents" in body and "total" in body)

    # --- Détail d'un document existant (si la base n'est pas vide) ---
    first_id: int | None = None
    if resp.status_code == 200 and resp.json().get("total", 0) > 0:
        first_id = resp.json()["documents"][0]["id"]
        resp_detail = http("GET", f"/api/v1/documents/{first_id}", headers=JSON_HEADERS)
        show(f"documents détail id={first_id}", resp_detail)
        assert_test(f"documents détail id={first_id} → 200",
                    resp_detail.status_code == 200)
    else:
        warn("aucun document en base → test du détail ignoré")

    # --- Document inexistant → 404 ---
    resp = http("GET", "/api/v1/documents/999999", headers=JSON_HEADERS)
    show("documents inexistant", resp)
    assert_test("documents id=999999 → 404", resp.status_code == 404)

    # --- ID invalide (non entier) → 422 ---
    resp = http("GET", "/api/v1/documents/abc", headers=JSON_HEADERS)
    show("documents id invalide", resp)
    assert_test("documents id='abc' → 422", resp.status_code == 422)

    # --- Suppression d'un document inexistant → 404 ---
    resp = http("DELETE", "/api/v1/documents/999999", headers=JSON_HEADERS)
    show("documents delete inexistant", resp)
    assert_test("DELETE id=999999 → 404", resp.status_code == 404)


# =====================================================================
# Section 5 : Retrieve — POST /api/v1/rag/retrieve (auth requise)
# =====================================================================

def test_retrieve() -> None:
    section("SECTION 5 : RETRIEVE — POST /api/v1/rag/retrieve")

    # --- Cas nominal ---
    payload = {"query": "pompe vibration", "top_k": 5}
    resp = http("POST", "/api/v1/rag/retrieve", headers=JSON_HEADERS, json=payload)
    show("retrieve simple", resp)
    assert_test("retrieve simple → 200", resp.status_code == 200)
    if resp.status_code == 200:
        body = resp.json()
        assert_test("retrieve → clés results/strategy_name présentes",
                    "results" in body and "strategy_name" in body)

    # --- Avec filtres ---
    payload = {
        "query": "pompe vibration",
        "top_k": 5,
        "filters": {"source_type": "panne", "min_score": 0.5},
    }
    resp = http("POST", "/api/v1/rag/retrieve", headers=JSON_HEADERS, json=payload)
    show("retrieve avec filtres", resp)
    assert_test("retrieve avec filtres → 200", resp.status_code == 200)

    # --- Requête vide → 422 (min_length=1) ---
    resp = http("POST", "/api/v1/rag/retrieve", headers=JSON_HEADERS,
                json={"query": "", "top_k": 5})
    show("retrieve query vide", resp)
    assert_test("retrieve query vide → 422", resp.status_code == 422)

    # --- top_k=0 → 422 (ge=1) ---
    resp = http("POST", "/api/v1/rag/retrieve", headers=JSON_HEADERS,
                json={"query": "pompe", "top_k": 0})
    show("retrieve top_k=0", resp)
    assert_test("retrieve top_k=0 → 422", resp.status_code == 422)

    # --- top_k=100 → 422 (le=50) ---
    resp = http("POST", "/api/v1/rag/retrieve", headers=JSON_HEADERS,
                json={"query": "pompe", "top_k": 100})
    show("retrieve top_k=100", resp)
    assert_test("retrieve top_k=100 → 422", resp.status_code == 422)

    # --- Sans auth → 422 ---
    resp = http("POST", "/api/v1/rag/retrieve",
                headers={"Content-Type": "application/json"},
                json={"query": "pompe vibration"})
    show("retrieve sans auth", resp)
    assert_test("retrieve sans auth → 422", resp.status_code == 422)


# =====================================================================
# Section 6 : Rerank — POST /api/v1/rag/rerank (auth requise)
# =====================================================================

def test_rerank() -> None:
    section("SECTION 6 : RERANK — POST /api/v1/rag/rerank")

    # --- Étape préalable : récupérer des candidats via /retrieve ---
    resp = http("POST", "/api/v1/rag/retrieve", headers=JSON_HEADERS,
                json={"query": "pompe vibration", "top_k": 5})
    candidates: list[dict] = []
    if resp.status_code == 200:
        candidates = resp.json().get("results", [])
    if not candidates:
        warn("aucun chunk récupéré → le test de rerank nominal sera sauté")
    else:
        payload = {"query": "pompe vibration", "candidates": candidates, "top_k": 3}
        resp_rr = http("POST", "/api/v1/rag/rerank", headers=JSON_HEADERS,
                       json=payload, timeout=TIMEOUT_LONG)
        show(f"rerank de {len(candidates)} candidats", resp_rr)
        assert_test("rerank candidats réels → 200", resp_rr.status_code == 200)

    # --- Candidats vides → 422 (min_length=1) ---
    resp = http("POST", "/api/v1/rag/rerank", headers=JSON_HEADERS,
                json={"query": "pompe", "candidates": []})
    show("rerank candidats vides", resp)
    assert_test("rerank candidats vides → 422", resp.status_code == 422)

    # --- Query vide → 422 (min_length=1) ---
    resp = http("POST", "/api/v1/rag/rerank", headers=JSON_HEADERS,
                json={"query": "", "candidates": candidates or [{
                    "chunk_id": "x:1", "content": "contenu", "score": 0.9,
                    "rank": 1, "source_name": "x.pdf", "source_type": "document",
                }]})
    show("rerank query vide", resp)
    assert_test("rerank query vide → 422", resp.status_code == 422)

    # --- Champs manquants (pas de candidates) → 422 ---
    resp = http("POST", "/api/v1/rag/rerank", headers=JSON_HEADERS,
                json={"query": "pompe"})
    show("rerank champs manquants", resp)
    assert_test("rerank champs manquants → 422", resp.status_code == 422)


# =====================================================================
# Section 7 : Search — POST /api/v1/rag/search (pipeline complet)
# =====================================================================

def test_search() -> None:
    section("SECTION 7 : SEARCH — POST /api/v1/rag/search")

    # --- Pipeline complet : retrieve → rerank → generate ---
    payload = {"query": "Pourquoi la pompe vibre-t-elle ?", "top_k": 5,
               "rerank": True, "generate": True}
    start = time.perf_counter()
    resp = http("POST", "/api/v1/rag/search", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    elapsed = time.perf_counter() - start
    show("search pipeline complet", resp, max_body=500)
    print(f"{DIM}    Durée client  : {elapsed:.2f} s{RESET}")

    assert_test("search pipeline complet → 200", resp.status_code == 200)
    if resp.status_code == 200:
        body = resp.json()
        assert_test(
            "search → clés answer/strategy_info/duration_ms/llm_error présentes",
            all(k in body for k in
                ("answer", "strategy_info", "duration_ms", "llm_error")),
        )
        # Une erreur LLM (quota, réseau…) n'est pas fatale : on avertit seulement.
        if body.get("llm_error"):
            warn(f"llm_error renseigné (dégradation gracieuse) : {body['llm_error']}")

    # --- Sans rerank (rerank=false, generate=true) ---
    payload = {"query": "pompe vibration", "top_k": 3,
               "rerank": False, "generate": True}
    resp = http("POST", "/api/v1/rag/search", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    show("search sans rerank", resp, max_body=300)
    assert_test("search rerank=false → 200", resp.status_code == 200)

    # --- Sans génération (rerank=true, generate=false) → answer vide ---
    payload = {"query": "pompe vibration", "top_k": 3,
               "rerank": True, "generate": False}
    resp = http("POST", "/api/v1/rag/search", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    show("search sans génération", resp, max_body=300)
    assert_test("search generate=false → 200", resp.status_code == 200)
    if resp.status_code == 200:
        assert_test("search generate=false → answer == ''",
                    resp.json().get("answer") == "")

    # --- Avec filtres ---
    payload = {"query": "pompe vibration", "top_k": 3,
               "filters": {"source_type": "panne"}}
    resp = http("POST", "/api/v1/rag/search", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    show("search avec filtres", resp, max_body=300)
    assert_test("search filtre source_type=panne → 200", resp.status_code == 200)

    # --- Query vide → 422 ---
    resp = http("POST", "/api/v1/rag/search", headers=JSON_HEADERS,
                json={"query": ""})
    show("search query vide", resp)
    assert_test("search query vide → 422", resp.status_code == 422)

    # --- Sans auth → 422 ---
    resp = http("POST", "/api/v1/rag/search",
                headers={"Content-Type": "application/json"},
                json={"query": "pompe vibration"})
    show("search sans auth", resp)
    assert_test("search sans auth → 422", resp.status_code == 422)

    # --- top_k=-1 → 422 (ge=1) ---
    resp = http("POST", "/api/v1/rag/search", headers=JSON_HEADERS,
                json={"query": "pompe", "top_k": -1})
    show("search top_k=-1", resp)
    assert_test("search top_k=-1 → 422", resp.status_code == 422)


# =====================================================================
# Section 8 : Ingest file — POST /api/v1/ingest/file (multipart)
# =====================================================================

def test_ingest_file() -> None:
    section("SECTION 8 : INGEST FILE — POST /api/v1/ingest/file")

    # --- Préparation du fichier de test (contenu maintenance réaliste) ---
    TMP_FILE_1.write_text(
        "Rapport de maintenance pompe P-101\n"
        "La pompe présente des vibrations anormales depuis le 12 mars.\n"
        "Analyse : déséquilibre du rotor et usure des paliers côté accouplement.\n"
        "Action corrective : remplacement des paliers et équilibrage dynamique.\n"
        "Recommandation : contrôle vibratoire mensuel à 1500 tr/min.\n",
        encoding="utf-8",
    )

    def upload(path: Path, data: dict | None = None,
               headers: dict | None = AUTH_HEADERS) -> requests.Response | _FailedResponse:
        """Upload multipart : uniquement l'en-tête Authorization (requests
        doit définir lui-même le Content-Type avec boundary)."""
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "text/plain")}
            return http("POST", "/api/v1/ingest/file", headers=headers,
                        files=files, data=data, timeout=TIMEOUT_LONG)

    # --- Upload avec métadonnées (id_equipement + chunk_size personnalisé) ---
    resp = upload(TMP_FILE_1, data={"id_equipement": "42", "chunk_size": "300"})
    show("ingest file avec métadonnées", resp)
    assert_test("ingest file avec métadonnées → 200", resp.status_code == 200)

    # --- Upload sans métadonnées (valeurs par défaut du pipeline) ---
    resp = upload(TMP_FILE_1)
    show("ingest file sans métadonnées", resp)
    assert_test("ingest file sans métadonnées → 200", resp.status_code == 200)

    # --- Upload sans auth → 401 (token invalide) ou 422 (header absent) ---
    resp = upload(TMP_FILE_1, headers={})
    show("ingest file sans auth", resp)
    assert_test("ingest file sans auth → 401 ou 422",
                resp.status_code in (401, 422))

    # --- Upload d'un fichier vide → 200 (erreur pipeline capturée dans le JSON) ---
    empty_file = Path("/tmp/test_manual_api_empty.txt")
    empty_file.write_text("", encoding="utf-8")
    resp = upload(empty_file)
    show("ingest file vide", resp)
    assert_test("ingest file vide → 200 (erreur capturée)",
                resp.status_code == 200)
    empty_file.unlink(missing_ok=True)


# =====================================================================
# Section 9 : Ingest database — POST /api/v1/ingest/database
# =====================================================================

def test_ingest_database() -> None:
    section("SECTION 9 : INGEST DATABASE — POST /api/v1/ingest/database")

    db = load_db_config()

    # --- Ingestion de la table "panne" (base locale gmao_rag) ---
    payload = {
        "host": db["host"],
        "port": db["port"],
        "database": db["database"],
        "user": db["user"],
        "password": db["password"],
        "table": "panne",
    }
    resp = http("POST", "/api/v1/ingest/database", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    show(f"ingest database table=panne ({db['database']})", resp)
    assert_test("ingest database table=panne → 200", resp.status_code == 200)

    # --- Requête SQL personnalisée (override de "table") ---
    payload_sql = dict(payload)
    payload_sql["query"] = (
        "SELECT id_panne, description FROM panne ORDER BY id_panne DESC LIMIT 20"
    )
    resp = http("POST", "/api/v1/ingest/database", headers=JSON_HEADERS,
                json=payload_sql, timeout=TIMEOUT_LONG)
    show("ingest database SQL personnalisée", resp)
    assert_test("ingest database SQL personnalisée → 200", resp.status_code == 200)

    # --- Host vide → 422 (min_length=1) ---
    bad = dict(payload, host="")
    resp = http("POST", "/api/v1/ingest/database", headers=JSON_HEADERS,
                json=bad, timeout=TIMEOUT_LONG)
    show("ingest database host vide", resp)
    assert_test("ingest database host vide → 422", resp.status_code == 422)

    # --- Port négatif → 422 (gt=0) ---
    bad = dict(payload, port=-1)
    resp = http("POST", "/api/v1/ingest/database", headers=JSON_HEADERS,
                json=bad, timeout=TIMEOUT_LONG)
    show("ingest database port négatif", resp)
    assert_test("ingest database port=-1 → 422", resp.status_code == 422)

    # --- Sans auth → 422 ---
    resp = http("POST", "/api/v1/ingest/database",
                headers={"Content-Type": "application/json"},
                json=payload, timeout=TIMEOUT_LONG)
    show("ingest database sans auth", resp)
    assert_test("ingest database sans auth → 422", resp.status_code == 422)


# =====================================================================
# Section 10 : Ingest files — POST /api/v1/ingest/files (batch)
# =====================================================================

def test_ingest_files() -> None:
    section("SECTION 10 : INGEST FILES — POST /api/v1/ingest/files")

    # --- Fichier n°2 pour le test multi-fichiers ---
    TMP_FILE_2.write_text(
        "Procédure de graissage moteur M-200\n"
        "Graisser les roulements toutes les 500 heures avec de la graisse EP2.\n"
        "Vérifier la température des paliers après chaque graissage.\n",
        encoding="utf-8",
    )

    # --- Un seul fichier ---
    payload = {"paths": [str(TMP_FILE_1)]}
    resp = http("POST", "/api/v1/ingest/files", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    show("ingest files ×1", resp)
    assert_test("ingest files ×1 → 200", resp.status_code == 200)
    if resp.status_code == 200:
        assert_test("ingest files ×1 → total_files == 1",
                    resp.json().get("total_files") == 1)

    # --- Deux fichiers ---
    payload = {"paths": [str(TMP_FILE_1), str(TMP_FILE_2)]}
    resp = http("POST", "/api/v1/ingest/files", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    show("ingest files ×2", resp)
    assert_test("ingest files ×2 → 200", resp.status_code == 200)

    # --- Liste de chemins vide → 422 (min_length=1) ---
    resp = http("POST", "/api/v1/ingest/files", headers=JSON_HEADERS,
                json={"paths": []}, timeout=TIMEOUT_LONG)
    show("ingest files chemins vides", resp)
    assert_test("ingest files paths=[] → 422", resp.status_code == 422)

    # --- Chemin inexistant → 200 (l'erreur est capturée par fichier) ---
    payload = {"paths": ["/tmp/fichier_qui_n_existe_pas_12345.txt"]}
    resp = http("POST", "/api/v1/ingest/files", headers=JSON_HEADERS,
                json=payload, timeout=TIMEOUT_LONG)
    show("ingest files chemin inexistant", resp)
    assert_test("ingest files chemin inexistant → 200 (erreur capturée)",
                resp.status_code == 200)


# =====================================================================
# Section 11 : Résumé final
# =====================================================================

def cleanup() -> None:
    """Supprime les fichiers temporaires créés par les tests."""
    for tmp in (TMP_FILE_1, TMP_FILE_2):
        tmp.unlink(missing_ok=True)


def main() -> None:
    start = time.perf_counter()

    print(f"{CYAN}Tests manuels de l'API GMAO-RAG — {BASE_URL}{RESET}")
    print(f"{DIM}Serveur cible : {BASE_URL} | Clé API : {'*' * len(API_KEY)}{RESET}")

    test_health()
    test_strategies()
    test_stats()
    test_documents()
    test_retrieve()
    test_rerank()
    test_search()
    test_ingest_file()
    test_ingest_database()
    test_ingest_files()

    cleanup()

    elapsed = time.perf_counter() - start
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"  Durée totale : {elapsed:.1f} s")
    print(f"{'=' * 60}")

    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
