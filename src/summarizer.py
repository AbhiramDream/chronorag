"""
summarizer.py
Lightweight extractive summarizer using word frequency scoring.
No external APIs — works entirely offline.
"""

import re
from collections import Counter

# Common stop words to ignore during scoring
STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "and",
    "or", "for", "of", "with", "that", "this", "was", "are", "be",
    "have", "i", "you", "he", "she", "we", "they", "do", "not", "so",
    "but", "as", "if", "my", "your", "his", "her", "its", "our"
}


def generate_summary(text: str, max_sentences: int = 3) -> str:
    """
    Extracts the top `max_sentences` most informative sentences
    from the given text using term frequency scoring.
    """
    if not text or not text.strip():
        return ""

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        # If no sentence boundaries, just return a truncated version
        return text[:300].strip()

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Count word frequencies (excluding stop words)
    words = re.findall(r"\b\w+\b", text.lower())
    freq = Counter(w for w in words if w not in STOP_WORDS and len(w) > 2)

    # Score each sentence by sum of its word frequencies
    def score(sentence: str) -> float:
        s_words = re.findall(r"\b\w+\b", sentence.lower())
        return sum(freq.get(w, 0) for w in s_words if w not in STOP_WORDS)

    ranked = sorted(sentences, key=score, reverse=True)
    top = ranked[:max_sentences]

    # Return in original order for readability
    ordered = [s for s in sentences if s in top]
    return " ".join(ordered)