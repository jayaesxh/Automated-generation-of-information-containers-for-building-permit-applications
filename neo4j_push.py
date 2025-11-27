# neo4j_push.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

# We reuse the same .env loader as neo4j_loader for consistency
try:
    from neo4j_loader import _load_env
except ImportError:
    _load_env = None  # will handle below


def _ensure_env():
    """
    Ensure Neo4j env vars are loaded, using neo4j_loader._load_env if available,
    otherwise do a minimal inline .env load from repo root.
    """
    if _load_env is not None:
        _load_env()
        return

    # Fallback: simple local .env loader
    root = Path(__file__).resolve().parent
    for env_path in (root / ".env", root.parent / ".env"):
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
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception as e:
            print(f"[neo4j_push] Failed to read {env_path}: {e}")


def _get_driver():
    _ensure_env()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not (uri and user and password):
        raise RuntimeError(
            "[neo4j_push] Neo4j credentials are not set. "
            "Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (and optional NEO4J_DB)."
        )

    return GraphDatabase.driver(uri, auth=(user, password))


def _load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def push_case_to_neo4j(case_id: str, rag_dir: Path) -> None:
    """
    Populate a simple OntoBPR-style graph in Neo4j:

    (BuildingApplication)-[:HAS_DOCUMENT]->(Document)-[:HAS_CHUNK]->(Chunk)
    """
    chunks_path = rag_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"[neo4j_push] chunks.jsonl missing at {chunks_path}, skipping Neo4j push.")
        return

    chunks = _load_chunks(chunks_path)
    if not chunks:
        print(f"[neo4j_push] No chunks for case {case_id}, skipping Neo4j push.")
        return

    base_iri = f"https://example.org/{case_id}#"
    ba_iri = f"{base_iri}BA_{case_id}"

    driver = _get_driver()
    neo4j_db = os.getenv("NEO4J_DB", "neo4j")

    with driver.session(database=neo4j_db) as session:
        # 1) Clear existing nodes for this case
        session.run("MATCH (n {case_id: $cid}) DETACH DELETE n", cid=case_id)

        # 2) Create BuildingApplication node
        session.run(
            """
            CREATE (ba:BuildingApplication:OntoBPR {
                iri: $iri,
                case_id: $cid,
                name: $name
            })
            """,
            iri=ba_iri,
            cid=case_id,
            name=f"Building Application {case_id}",
        )

        # 3) Documents + chunks
        for ch in chunks:
            doc_iri = ch.get("doc_iri")
            doc_name = ch.get("name") or "Unknown document"
            chunk_id = int(ch.get("chunk_id", 0))
            text = (ch.get("text") or "").strip()

            if not doc_iri:
                continue

            # Document node + relation to BA
            session.run(
                """
                MATCH (ba:BuildingApplication {case_id: $cid})
                MERGE (d:Document {iri: $doc_iri, case_id: $cid})
                  ON CREATE SET d.name = $doc_name
                MERGE (ba)-[:HAS_DOCUMENT]->(d)
                """,
                cid=case_id,
                doc_iri=doc_iri,
                doc_name=doc_name,
            )

            # Chunk node + relation to Document
            if text:
                session.run(
                    """
                    MATCH (d:Document {iri: $doc_iri, case_id: $cid})
                    MERGE (c:Chunk {case_id: $cid, chunk_id: $chunk_id})
                      ON CREATE SET c.text = $text
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    cid=case_id,
                    doc_iri=doc_iri,
                    chunk_id=chunk_id,
                    text=text,
                )

    driver.close()
    print(f"[neo4j_push] KG populated for case {case_id} ({len(chunks)} chunks).")
