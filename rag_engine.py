"""
Hybrid RAG Module for ICDD Pipeline.
Strategy: PRIORITIZE SentenceTransformers (Semantic Search).

Analysis of previous runs showed LightRAG failing to extract entities 
(0 entities/relations) due to local LLM limitations, while generating 
excessive logs.

This module now defaults to robust Semantic Search (SBERT).
"""
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import logging
import warnings

# ============================================================================
# Configuration & Logging
# ============================================================================
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Nuclear option for logging suppression
logging.getLogger().setLevel(logging.ERROR)
_SILENCE = [
    "lightrag", "nano-vectordb", "httpx", "urllib3", "transformers", 
    "sentence_transformers", "chromadb", "faiss"
]
for log_name in _SILENCE:
    l = logging.getLogger(log_name)
    l.setLevel(logging.CRITICAL)
    l.propagate = False
    l.disabled = True

# ============================================================================
# Imports
# ============================================================================
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

# Only import LightRAG if absolutely needed (fallback)
try:
    import nest_asyncio
    nest_asyncio.apply()
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    _HAS_LIGHTRAG = True
except ImportError:
    _HAS_LIGHTRAG = False

_EMBED_DIM = 384

# ============================================================================
# Helper Functions
# ============================================================================
def _load_chunks(rag_dir: Path) -> List[Dict[str, Any]]:
    chunks_file = rag_dir / "chunks.jsonl"
    if not chunks_file.exists():
        return []
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunks.append(json.loads(line.strip()))
            except:
                pass
    return chunks

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# ============================================================================
# Strategy A: Sentence Transformers (PRIMARY)
# ============================================================================
def _build_sbert_index(case_id: str, rag_dir: Path, chunks: List[Dict]) -> None:
    if not _HAS_SBERT:
        print("[RAG] sentence-transformers not installed.")
        return
    
    texts = [c.get("text", "").strip() for c in chunks if c.get("text", "").strip()]
    if not texts:
        return

    print(f"[RAG] Building semantic index (SBERT) for {case_id}...")
    store_dir = rag_dir / "semantic_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        np.savez_compressed(store_dir / "embeddings.npz", embeddings=embeddings)
        with open(store_dir / "chunks.json", "w") as f:
            json.dump(chunks, f)
        print(f"[RAG] ✓ Semantic index built")
    except Exception as e:
        print(f"[RAG] SBERT build failed: {e}")

def _retrieve_sbert(case_id: str, rag_dir: Path, query: str, top_k: int) -> str:
    store_dir = rag_dir / "semantic_store"
    if not (store_dir / "embeddings.npz").exists():
        return ""
    
    try:
        data = np.load(store_dir / "embeddings.npz")
        embeddings = data['embeddings']
        with open(store_dir / "chunks.json") as f:
            chunks = json.load(f)
            
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        
        # Cosine similarity
        norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        norm_query = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        
        scores = np.dot(norm_emb, norm_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_idx:
            txt = chunks[i].get("text", "").strip()
            src = chunks[i].get("source", "unknown")
            if txt:
                results.append(f"[{src}] {txt}")
        
        return "\n\n".join(results)
    except Exception:
        return ""

# ============================================================================
# Strategy B: LightRAG (FALLBACK / LEGACY)
# ============================================================================
async def _async_hash_embedding(texts: List[str]) -> List[List[float]]:
    def _sync_hash(texts_list):
        arr = np.zeros((len(texts_list), _EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts_list):
            if not t: continue
            for tok in t.split():
                h = hash(tok) % _EMBED_DIM
                arr[i, h] += 1.0
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        return (arr / norms).tolist()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_hash, texts)

async def _build_lightrag_async(work_dir: Path, texts: List[str]) -> None:
    async def _dummy_llm(prompt, **kwargs):
        return json.dumps({"keywords": [], "content": "Dummy"})
    
    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(_EMBED_DIM, 8192, _async_hash_embedding),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.ainsert(texts)

async def _query_lightrag_async(work_dir: Path, query: str, top_k: int) -> str:
    async def _dummy_llm(prompt, **kwargs):
        return json.dumps({"keywords": [], "content": "Dummy"})
    
    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(_EMBED_DIM, 8192, _async_hash_embedding),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    result = await rag.aquery(query, param=QueryParam(mode="local", top_k=top_k))
    return str(result) if result else ""

# ============================================================================
# Main API
# ============================================================================
def build_lightrag_index(case_id: str, rag_dir: Path, store_dir: Optional[Path] = None) -> None:
    chunks = _load_chunks(rag_dir)
    if not chunks:
        return

    # 1. Prioritize SBERT (Robust, Clean)
    if _HAS_SBERT:
        _build_sbert_index(case_id, rag_dir, chunks)
        return

    # 2. Fallback to LightRAG (Only if SBERT missing)
    if _HAS_LIGHTRAG:
        try:
            if store_dir is None: store_dir = rag_dir / "lightrag_store"
            texts = [c.get("text", "").strip() for c in chunks if c.get("text", "").strip()]
            print(f"[RAG] Attempting LightRAG build (Fallback)...")
            _run_async(_build_lightrag_async(store_dir, texts))
            print(f"[RAG] ✓ LightRAG index built")
        except Exception:
            pass

def retrieve_context_for_case(case_id: str, rag_dir: Path, query: str, top_k: int = 20) -> str:
    # 1. Prioritize SBERT
    if _HAS_SBERT:
        result = _retrieve_sbert(case_id, rag_dir, query, top_k)
        if result:
            print(f"[RAG] ✓ Context retrieved via Semantic Search")
            return result

    # 2. Fallback to LightRAG
    if _HAS_LIGHTRAG:
        store_dir = rag_dir / "lightrag_store"
        if store_dir.exists():
            try:
                result = _run_async(_query_lightrag_async(store_dir, query, top_k))
                if result:
                    print(f"[RAG] ✓ Context retrieved via LightRAG")
                    return result
            except Exception:
                pass
            
    return ""

def ensure_lightrag_index(case_id: str, rag_dir: Path, store_dir: Optional[Path] = None) -> None:
    return build_lightrag_index(case_id, rag_dir, store_dir)
