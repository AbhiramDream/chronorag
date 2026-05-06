"""
vector_store.py
Kept for backwards compatibility.
The main retrieval logic is in rag_engine.py.
"""


def build_faiss(messages: list[dict]):
    """Returns texts for indexing. Actual indexing done in rag_engine."""
    texts = [m["text"] for m in messages if m.get("text", "").strip()]
    return None, texts