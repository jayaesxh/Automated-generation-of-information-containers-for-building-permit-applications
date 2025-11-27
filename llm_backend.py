from __future__ import annotations

import os
from typing import List, Dict, Optional

from huggingface_hub import InferenceClient

# ============================================================
# Configuration
# ============================================================

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
_CLIENT: Optional[InferenceClient] = None


def _get_client() -> InferenceClient:
    """Lazily create and cache InferenceClient."""
    global _CLIENT

    if _CLIENT is not None:
        return _CLIENT

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "[llm_backend] HF_TOKEN not set. Set it before running:\n"
            "  import os; os.environ['HF_TOKEN'] = 'hf_your_token_here'"
        )

    _CLIENT = InferenceClient(model=HF_MODEL_ID, token=token)
    print(f"[llm_backend] Using HF API with model={HF_MODEL_ID}")
    return _CLIENT


def call_llm(messages: List[Dict[str, str]]) -> str:
    """Call HF Inference API."""
    client = _get_client()

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as e:
        raise RuntimeError(f"[llm_backend] API call failed: {e}") from e

    try:
        choice = response.choices[0]
        msg = choice.message
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
    except Exception:
        content = str(response)

    return content.strip()