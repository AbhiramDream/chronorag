"""
persona_extractor.py
Extracts structured user persona ONLY from actual conversational evidence.
Uses rule-based keyword pattern matching — no hallucination.
"""

import json
import os
import re

PERSONA_DIR = "persona"

# ──────────────────────────────────────────────
# Keyword maps — each key maps to the evidence label
# ──────────────────────────────────────────────

HABIT_PATTERNS = {
    r"\b(sleep|slept|nap|tired|sleepy|wake|woke|bed)\b": "Late/irregular sleeper",
    r"\b(food|eat|ate|hungry|lunch|dinner|breakfast|snack|meal|cook|pizza|burger)\b": "Discusses food frequently",
    r"\b(gym|workout|exercise|run|jog|lift|fitness|training)\b": "Fitness-conscious",
    r"\b(coffee|tea|chai)\b": "Regular hot-drink consumer",
    r"\b(game|gaming|play|ps5|xbox|pc game)\b": "Gamer",
    r"\b(read|book|novel|reading)\b": "Reader",
}

TRAIT_PATTERNS = {
    r"\b(haha|lol|lmao|hehe|funny|joke|jokes|laugh|hilarious)\b": "Humorous",
    r"\b(sad|cry|depressed|upset|miss|lonely|heartbroken)\b": "Emotional",
    r"\b(love|care|sweet|kind|help|support)\b": "Caring",
    r"\b(stress|stressed|anxious|worry|worried|nervous|panic)\b": "Anxious",
    r"\b(sure|definitely|absolutely|confident|believe)\b": "Confident",
    r"\b(idk|maybe|not sure|confused|unsure)\b": "Indecisive",
}

PERSONAL_FACT_PATTERNS = {
    r"\b(sister|brother|mom|dad|parent|family|cousin|uncle|aunt)\b": "Has family members mentioned",
    r"\b(boyfriend|girlfriend|partner|wife|husband)\b": "In a relationship",
    r"\b(college|university|exam|semester|course|degree|study|student)\b": "Student",
    r"\b(job|work|office|boss|salary|internship|interview)\b": "Working/job-seeking",
    r"\b(dog|cat|pet)\b": "Has a pet",
}

EMOJI_SAMPLE = ["😂", "😭", "❤️", "🔥", "😅", "🥺", "😍", "💀", "🙏", "😤"]


def _match_patterns(text: str, patterns: dict) -> set:
    """Returns the set of labels where any keyword is found in text."""
    found = set()
    lower = text.lower()
    for pattern, label in patterns.items():
        if re.search(pattern, lower):
            found.add(label)
    return found


def extract_persona(messages: list[dict]) -> dict:
    """
    Scans all messages for conversational evidence and builds a persona JSON.
    Deduplicates all categories before saving.

    Returns the persona dict and saves to persona/persona.json.
    """
    os.makedirs(PERSONA_DIR, exist_ok=True)

    habits: set = set()
    traits: set = set()
    personal_facts: set = set()

    emoji_count = 0
    short_msg_count = 0
    question_count = 0
    total = len(messages)

    for msg in messages:
        text = msg.get("text", "")
        if not text:
            continue

        habits.update(_match_patterns(text, HABIT_PATTERNS))
        traits.update(_match_patterns(text, TRAIT_PATTERNS))
        personal_facts.update(_match_patterns(text, PERSONAL_FACT_PATTERNS))

        # Communication style signals
        for e in EMOJI_SAMPLE:
            if e in text:
                emoji_count += 1
                break  # count once per message

        word_count = len(text.split())
        if word_count <= 5:
            short_msg_count += 1

        if "?" in text:
            question_count += 1

    # Derive communication style from counts
    emoji_usage = (
        "frequent" if emoji_count > total * 0.15
        else "moderate" if emoji_count > total * 0.05
        else "low"
    )
    tone = "casual" if short_msg_count > total * 0.4 else "conversational"

    persona = {
        "habits": sorted(habits),
        "personality_traits": sorted(traits),
        "personal_facts": sorted(personal_facts),
        "communication_style": {
            "short_messages": short_msg_count > total * 0.4,
            "emoji_usage": emoji_usage,
            "tone": tone,
            "asks_questions_frequently": question_count > total * 0.2,
        }
    }

    path = os.path.join(PERSONA_DIR, "persona.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)

    return persona