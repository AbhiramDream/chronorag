"""
checkpoint_manager.py
Creates two types of checkpoints:
1. Topic checkpoints  — one per detected topic
2. Message checkpoints — one every 100 chronological messages
"""

import json
import os
from src.summarizer import generate_summary

CHECKPOINTS_DIR = "checkpoints"


def _ensure_dir():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def create_topic_checkpoints(topics: list[dict]) -> list[dict]:
    """
    For each detected topic, generate a summary of its messages
    and write to topic_checkpoints.json.

    Returns the list of checkpoint dicts.
    """
    _ensure_dir()
    checkpoints = []

    for topic in topics:
        msgs = topic["messages"]
        if not msgs:
            continue

        # Combine first 60 messages for the summary (keep it manageable)
        combined = " ".join(m["text"] for m in msgs[:60])
        summary = generate_summary(combined, max_sentences=3)

        checkpoints.append({
            "topic_id": topic["topic_id"],
            "start_msg": msgs[0]["msg_id"],
            "end_msg":   msgs[-1]["msg_id"],
            "message_count": len(msgs),
            "summary": summary
        })

    path = os.path.join(CHECKPOINTS_DIR, "topic_checkpoints.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2, ensure_ascii=False)

    return checkpoints


def create_message_checkpoints(messages: list[dict]) -> list[dict]:
    """
    Every 100 messages, generate a summary and write to message_checkpoints.json.
    These are independent of topic boundaries.

    Returns the list of checkpoint dicts.
    """
    _ensure_dir()
    checkpoints = []

    for i in range(0, len(messages), 100):
        chunk = messages[i: i + 100]
        combined = " ".join(m["text"] for m in chunk[:60])
        summary = generate_summary(combined, max_sentences=3)

        checkpoints.append({
            "start_msg": chunk[0]["msg_id"],
            "end_msg":   chunk[-1]["msg_id"],
            "message_count": len(chunk),
            "summary": summary
        })

    path = os.path.join(CHECKPOINTS_DIR, "message_checkpoints.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2, ensure_ascii=False)

    return checkpoints