# duel_retriever.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from vector_index import build_vector_index, query_vector_index
from neo4j_loader import get_neo4j_context

# Assume repo structure: /workspace/icdd-rag-pipeline2
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def _infer_rag_dir(case_id: str) -> Path:
    """
    Best-effort guess for rag directory based on standard layout:
    output/<case_id>/rag/
    """
    return OUTPUT / case_id / "rag"


def _build_text_context(
    case_id: str,
    query: str,
    rag_dir: Path,
    top_k_text: int = 10,
) -> str:
    """
    Ensure vector index exists and return a formatted text context block.
    """
    rag_dir = Path(rag_dir)
    try:
        build_vector_index(case_id, rag_dir, persist=False)
    except Exception as e:
        print(f"[duel_retriever] build_vector_index failed: {e}")

    try:
        hits = query_vector_index(case_id, query, top_k=top_k_text)
    except Exception as e:
        print(f"[duel_retriever] query_vector_index failed: {e}")
        return ""

    parts: List[str] = []
    for i, t in enumerate(hits, start=1):
        t = (t or "").strip()
        if not t:
            continue
        parts.append(f"[TEXT HIT {i}]\n{t}")

    if not parts:
        return ""
    return "TEXT_CONTEXT:\n" + "\n\n".join(parts)


def _build_graph_context(
    case_id: str,
    max_graph_chunks: int = 30,
) -> str:
    """
    Fetch a KG-based context block from Neo4j using neo4j_loader.get_neo4j_context.
    """
    try:
        ctx = get_neo4j_context(case_id, max_chunks=max_graph_chunks)
    except Exception as e:
        print(f"[duel_retriever] get_neo4j_context failed: {e}")
        return ""

    if not ctx.strip():
        return ""
    return "GRAPH_CONTEXT:\n" + ctx.strip()


def build_context_for_case(
    case_id: str,
    query: str,
    rag_dir: Optional[Path] = None,
    top_k_text: int = 10,
    max_graph_chunks: int = 30,
) -> str:
    """
    Build a combined context string based on:
      - vector-based text retrieval over chunks.jsonl
      - simple KG context from Neo4j

    Parameters
    ----------
    case_id : str
        Case identifier (e.g. "Apl_25-04543-FULL").
    query : str
        Natural-language query describing what information you want to extract.
    rag_dir : Optional[Path]
        Path to output/<case_id>/rag. If None, it will be inferred.
    top_k_text : int
        Number of top text hits to include.
    max_graph_chunks : int
        Maximum number of KG context snippets to include.

    Returns
    -------
    str
        Multi-section context string with TEXT_CONTEXT and GRAPH_CONTEXT.
    """
    if rag_dir is None:
        rag_dir = _infer_rag_dir(case_id)

    text_block = _build_text_context(
        case_id=case_id,
        query=query,
        rag_dir=rag_dir,
        top_k_text=top_k_text,
    )
    graph_block = _build_graph_context(
        case_id=case_id,
        max_graph_chunks=max_graph_chunks,
    )

    blocks: List[str] = []
    if text_block.strip():
        blocks.append(text_block.strip())
    if graph_block.strip():
        blocks.append(graph_block.strip())

    if not blocks:
        return ""

    combined = "\n\n".join(blocks)
    print(
        f"[duel_retriever] Built dual context for {case_id}: "
        f"text_hits={top_k_text}, max_graph_snippets={max_graph_chunks}"
    )
    return combined
