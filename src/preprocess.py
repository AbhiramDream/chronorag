"""
preprocess.py
Loads the conversations CSV and splits each row into individual messages,
assigning chronological message IDs.
"""

import pandas as pd


def load_conversations(csv_path: str) -> list[dict]:
    """
    Reads a CSV where each row is one day's conversation.
    Splits each row into individual messages and assigns incremental IDs.

    Returns a list of dicts:
        {"msg_id": int, "text": str}
    """
    df = pd.read_csv(csv_path, header=None, on_bad_lines="skip")

    messages = []
    msg_id = 0

    for _, row in df.iterrows():
        cell = row[0]
        if pd.isna(cell):
            continue

        # Each row is one day; split by newline to get individual messages
        lines = str(cell).split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            msg_id += 1
            messages.append({
                "msg_id": msg_id,
                "text": line
            })

    return messages