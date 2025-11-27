# neo4j_loader.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from neo4j import GraphDatabase

# Project root (used to locate .env and output/)
ROOT = Path(__file__).resolve().parent


def _load_env() -> None:
    """
    Minimal .env loader so Neo4j credentials work inside Jupyter
    without manually exporting env vars.

    Looks for:
      - <repo_root>/.env
      - <repo_root>/../.env
    and sets os.environ[...] for each KEY=VALUE line if not already set.

    Also maps AURA_DB -> NEO4J_DB if the latter is not defined.
    """
    candidates = [
        ROOT / ".env",
        ROOT.parent / ".env",
    ]

    for env_path in candidates:
        if not env_path.exists():
            continue

        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Do not override variables already set in the environment
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception as e:
            print(f"[neo4j_loader] Failed to read {env_path}: {e}")

    # Map AURA_DB -> NEO4J_DB if needed
    if "NEO4J_DB" not in os.environ and "AURA_DB" in os.environ:
        os.environ["NEO4J_DB"] = os.environ["AURA_DB"]


def _get_driver():
    """
    Return a live Neo4j driver, after ensuring .env is loaded.
    """
    _load_env()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not (uri and user and password):
        raise RuntimeError(
            "[neo4j_loader] Neo4j credentials are not set. "
            "Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (and optional NEO4J_DB)."
        )

    return GraphDatabase.driver(uri, auth=(user, password))


def get_neo4j_context(case_id: str, max_chunks: int = 40) -> str:
    """
    Pull text back out of the KG as additional context for the LLM.

    Expects graph created by neo4j_push.push_case_to_neo4j:
      (BuildingApplication)-[:HAS_DOCUMENT]->(Document)-[:HAS_CHUNK]->(Chunk)
    """
    driver = _get_driver()
    neo4j_db = os.getenv("NEO4J_DB", os.getenv("AURA_DB", "neo4j"))

    with driver.session(database=neo4j_db) as session:
        result = session.run(
            """
            MATCH (ba:BuildingApplication {case_id: $cid})
                  -[:HAS_DOCUMENT]->(d:Document)
                  -[:HAS_CHUNK]->(c:Chunk)
            RETURN d.name   AS doc_name,
                   c.chunk_id AS chunk_id,
                   c.text    AS text
            ORDER BY d.name, c.chunk_id
            LIMIT $lim
            """,
            cid=case_id,
            lim=max_chunks,
        )
        rows = list(result)

    driver.close()

    if not rows:
        print(f"[neo4j_loader] No KG context for case {case_id}.")
        return ""

    parts: List[str] = []
    for r in rows:
        doc_name = r["doc_name"] or ""
        chunk_id = r["chunk_id"]
        text = (r["text"] or "").strip()
        if not text:
            continue
        parts.append(f"[{doc_name} | chunk {chunk_id}]\n{text}")

    ctx = "\n\n".join(parts)
    print(f"[neo4j_loader] Collected {len(parts)} KG-context snippets for case {case_id}.")
    return ctx
