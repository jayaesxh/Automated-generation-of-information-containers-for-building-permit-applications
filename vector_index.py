# vector_index.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

# In-memory index: case_id → {"texts": [...], "metas": [...], "embeddings": np.ndarray | None}
_INDEX: Dict[str, Dict[str, Any]] = {}
_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> Optional[SentenceTransformer]:
    """
    Lazily load the SentenceTransformer model if available.
    Returns None if sentence_transformers is not installed.
    """
    global _MODEL
    if not _HAS_ST:
        return None
    if _MODEL is None:
        # You can change the model name if needed
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _load_chunks(rag_dir: Path) -> List[Dict[str, Any]]:
    """
    Load chunks.jsonl from rag_dir and return list of dicts.
    """
    chunks_path = rag_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"[vector_index] No chunks.jsonl at {chunks_path}")

    chunks: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return chunks


def build_vector_index(
    case_id: str,
    rag_dir: Path,
    persist: bool = False,
) -> None:
    """
    Build an in-memory vector index for the given case_id from rag/chunks.jsonl.

    If sentence_transformers is available, this will compute embeddings.
    Otherwise it falls back to lexical scoring in query_vector_index().

    If persist=True and embeddings are available, also writes:
      - rag/vector_index.npz  (embeddings)
      - rag/vector_meta.json  (metadata for each chunk)
    """
    rag_dir = Path(rag_dir)
    chunks = _load_chunks(rag_dir)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        metas.append(
            {
                "doc_iri": c.get("doc_iri"),
                "name": c.get("name"),
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "role": c.get("role"),
            }
        )

    if not texts:
        print(f"[vector_index] No non-empty text chunks for case {case_id}.")
        _INDEX[case_id] = {"texts": [], "metas": [], "embeddings": None}
        return

    model = _get_model()
    if model is not None:
        # Compute embeddings
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
    else:
        emb = None

    _INDEX[case_id] = {
        "texts": texts,
        "metas": metas,
        "embeddings": emb,
    }

    if persist and emb is not None:
        np.savez(rag_dir / "vector_index.npz", embeddings=emb)
        (rag_dir / "vector_meta.json").write_text(
            json.dumps(metas, indent=2),
            encoding="utf-8",
        )

    print(
        f"[vector_index] Built index for case {case_id} "
        f"({len(texts)} chunks, embeddings={'yes' if emb is not None else 'no'})."
    )


def _cosine_scores(query_vec: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity scores between query_vec (d,) and emb (n,d).
    """
    if emb.size == 0:
        return np.zeros((0,), dtype="float32")

    # Normalize
    emb_norm = np.linalg.norm(emb, axis=1)
    q_norm = float(np.linalg.norm(query_vec))
    denom = emb_norm * q_norm
    denom = np.where(denom == 0.0, 1e-8, denom)
    scores = (emb @ query_vec) / denom
    return scores.astype("float32")


def query_vector_index(case_id: str, query: str, top_k: int = 10) -> List[str]:
    """
    Return top-k chunk texts for this case_id using embeddings (if available)
    or a simple lexical-scoring fallback if embeddings/model are not available.
    """
    data = _INDEX.get(case_id)
    if not data:
        print(
            f"[vector_index] No index for case {case_id}. "
            "Did you call build_vector_index()?"
        )
        return []

    texts: List[str] = data["texts"]
    emb = data["embeddings"]

    if not texts:
        return []

    model = _get_model()
    if emb is not None and model is not None:
        q_vec = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype("float32")
        scores = _cosine_scores(q_vec, emb)
        idx = np.argsort(-scores)[:top_k]
        return [texts[int(i)] for i in idx]

    # Lexical fallback
    q_tokens = [t.lower() for t in query.split() if t.strip()]
    if not q_tokens:
        return texts[:top_k]

    scored: List[Tuple[int, str]] = []
    for t in texts:
        lt = t.lower()
        score = sum(lt.count(tok) for tok in q_tokens)
        if score > 0:
            scored.append((score, t))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]
