"""
rag_engine.py
TF-IDF based retrieval engine.
Indexes message chunks AND topic summaries independently.
Retrieves the most relevant content for any query using cosine similarity.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# In-memory stores (populated by initialize_rag)
# ──────────────────────────────────────────────

_msg_texts: list[str] = []
_msg_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
_msg_matrix = None

_topic_summaries: list[str] = []
_topic_vectorizer = TfidfVectorizer(stop_words="english")
_topic_matrix = None

_msg_checkpoint_summaries: list[str] = []
_msg_cp_vectorizer = TfidfVectorizer(stop_words="english")
_msg_cp_matrix = None


def initialize_rag(messages: list[dict]) -> None:
    """
    Builds the TF-IDF indexes for:
    1. Raw message texts
    2. Topic checkpoint summaries
    3. Message (100-msg) checkpoint summaries
    """
    global _msg_texts, _msg_matrix
    global _topic_summaries, _topic_matrix
    global _msg_checkpoint_summaries, _msg_cp_matrix

    # --- Index 1: raw messages ---
    _msg_texts = [m["text"] for m in messages if m.get("text", "").strip()]
    if _msg_texts:
        _msg_matrix = _msg_vectorizer.fit_transform(_msg_texts)

    # --- Index 2: topic summaries ---
    try:
        with open("checkpoints/topic_checkpoints.json", encoding="utf-8") as f:
            topics = json.load(f)
        _topic_summaries = [t["summary"] for t in topics if t.get("summary", "").strip()]
        if _topic_summaries:
            _topic_matrix = _topic_vectorizer.fit_transform(_topic_summaries)
    except FileNotFoundError:
        pass

    # --- Index 3: message checkpoints ---
    try:
        with open("checkpoints/message_checkpoints.json", encoding="utf-8") as f:
            msg_cps = json.load(f)
        _msg_checkpoint_summaries = [c["summary"] for c in msg_cps if c.get("summary", "").strip()]
        if _msg_checkpoint_summaries:
            _msg_cp_matrix = _msg_cp_vectorizer.fit_transform(_msg_checkpoint_summaries)
    except FileNotFoundError:
        pass


def _top_k(query: str, vectorizer, matrix, texts: list[str], k: int) -> list[str]:
    """Retrieve top-k texts by cosine similarity to the query."""
    if matrix is None or not texts:
        return []
    try:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        idx = np.argsort(sims)[::-1][:k]
        return [texts[i] for i in idx if sims[i] > 0.01]
    except Exception:
        return []


def retrieve(query: str, k_msgs: int = 5, k_topics: int = 3, k_checkpoints: int = 2) -> dict:
    """
    Retrieves relevant content from all three indexes.

    Returns:
        {
            "message_chunks": [...],
            "topic_summaries": [...],
            "checkpoint_summaries": [...]
        }
    """
    return {
        "message_chunks":      _top_k(query, _msg_vectorizer, _msg_matrix, _msg_texts, k_msgs),
        "topic_summaries":     _top_k(query, _topic_vectorizer, _topic_matrix, _topic_summaries, k_topics),
        "checkpoint_summaries": _top_k(query, _msg_cp_vectorizer, _msg_cp_matrix, _msg_checkpoint_summaries, k_checkpoints),
    }