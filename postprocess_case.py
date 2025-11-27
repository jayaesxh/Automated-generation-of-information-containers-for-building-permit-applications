# postprocess_case.py
from __future__ import annotations

from pathlib import Path

from neo4j_loader import ROOT as ROOT_DIR, load_ontobpr_to_neo4j
from vector_index import build_vector_index


def run_for_case(case_id: str) -> None:
    """
    Run the *post-ICDD* steps for a given case:
      - Build vector index for chunks.jsonl
      - Load OntoBPR.ttl into Neo4j
    """
    out_dir = ROOT_DIR / "output" / case_id
    if not out_dir.exists():
        raise FileNotFoundError(f"[postprocess_case] Run directory not found: {out_dir}")

    # 1) Build vector index (text-level RAG)
    build_vector_index(case_id, out_dir)

    # 2) Load OntoBPR KG into Neo4j (graph-level RAG)
    load_ontobpr_to_neo4j(case_id, out_dir)

    print(f"[postprocess_case] Finished post-processing for case {case_id}.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python postprocess_case.py <CASE_ID>")
        raise SystemExit(1)

    run_for_case(sys.argv[1])
