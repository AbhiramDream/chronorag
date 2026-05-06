"""
topic_detector.py
Detects topic changes chronologically using TF-IDF + cosine similarity.

APPROACH:
- Fit TF-IDF ONCE across all messages (fast)
- Compare the CENTROID of the current topic's vectors against each new message
- When similarity drops below threshold AND topic is large enough → split
- This avoids re-fitting the vectorizer per message (which is O(N²) slow)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configurable parameters ───────────────────────────────────────────────────

# Similarity between running topic centroid and new message below this = topic shift
SIMILARITY_THRESHOLD = 0.05

# Minimum messages per topic before a split is allowed
MIN_TOPIC_LENGTH = 50

# How many recent messages to use for the running topic centroid
CENTROID_WINDOW = 30


def _clean(text: str) -> str:
    t = text.strip().lower()
    return t if len(t) >= 3 else ""


def detect_topics(messages: list[dict]) -> list[dict]:
    """
    Detects topic boundaries by comparing the running centroid of the
    current topic (TF-IDF average) against each new incoming message.

    All vectorization is done in a SINGLE fit_transform call for speed.

    Returns a list of topic dicts:
        {"topic_id": int, "messages": [msg, ...]}
    """
    # Filter empty messages
    valid = [(m, _clean(m.get("text", ""))) for m in messages]
    valid = [(m, t) for m, t in valid if t]

    if not valid:
        return []

    texts = [t for _, t in valid]

    # ── Fit TF-IDF once across ALL messages ──────────────────────────────────
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = vectorizer.fit_transform(texts)  # shape: (N, vocab)

    topics = []
    topic_id = 1
    current_indices: list[int] = []   # indices into matrix for current topic
    current_msgs: list[dict] = []

    for i in range(len(valid)):
        msg, _ = valid[i]
        current_indices.append(i)
        current_msgs.append(msg)

        # Only check for split after minimum topic length
        if len(current_msgs) < MIN_TOPIC_LENGTH:
            continue

        # Compute centroid of the last CENTROID_WINDOW message vectors (sparse mean)
        window_idx = current_indices[-CENTROID_WINDOW:]
        window_matrix = matrix[window_idx]
        # .mean() on sparse returns np.matrix — convert to array for sklearn
        centroid = np.asarray(window_matrix.mean(axis=0))

        # Compare centroid to next message (lookahead)
        next_i = i + 1
        if next_i >= len(valid):
            continue

        next_vec = matrix[next_i]
        sim = cosine_similarity(centroid, next_vec)[0][0]

        if sim < SIMILARITY_THRESHOLD:
            # Topic has shifted → finalize current topic
            topics.append({
                "topic_id": topic_id,
                "messages": current_msgs
            })
            topic_id += 1
            current_indices = []
            current_msgs = []

    # Append the final topic
    if current_msgs:
        topics.append({
            "topic_id": topic_id,
            "messages": current_msgs
        })

    return topics