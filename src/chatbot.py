"""
chatbot.py
Generates responses using:
1. RAG retrieval (message chunks + topic/checkpoint summaries)
2. Persona data
3. HuggingFace Inference API (Meta-Llama-3-8B-Instruct) via huggingface_hub
"""

import json
import os

from huggingface_hub import InferenceClient
from src.rag_engine import retrieve


def _get_hf_token() -> str | None:
    """Read HF_TOKEN from Streamlit secrets, environment, or .env file."""
    # Streamlit Cloud secrets (highest priority for deployment)
    try:
        import streamlit as st
        token = st.secrets.get("HF_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass
    # Local environment variable
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()
    # Local .env file
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1]
    except FileNotFoundError:
        pass
    return None


def _load_persona() -> str:
    """Load persona JSON as a compact string."""
    try:
        with open("persona/persona.json", encoding="utf-8") as f:
            persona = json.load(f)
        return json.dumps(persona, indent=2)
    except FileNotFoundError:
        return "{}"


def generate_response(query: str) -> str:
    """
    Builds a grounded prompt from RAG context + persona
    and sends it to Meta-Llama-3-8B-Instruct via HuggingFace.

    Falls back to a context dump if the API is unavailable.
    """
    # --- Retrieve relevant content ---
    results = retrieve(query)
    topic_ctx    = "\n".join(results["topic_summaries"])
    msg_ctx      = "\n".join(results["message_chunks"])
    cp_ctx       = "\n".join(results["checkpoint_summaries"])
    persona_str  = _load_persona()

    # --- Build prompt context ---
    context = f"""### Persona:
{persona_str}

### Relevant Topic Summaries:
{topic_ctx or 'None'}

### Relevant Checkpoint Summaries:
{cp_ctx or 'None'}

### Relevant Message Chunks:
{msg_ctx or 'None'}"""

    system_prompt = (
        "You are a helpful assistant that answers questions about a user "
        "based strictly on conversation data and extracted persona. "
        "Do not make up information. Be concise and specific."
    )

    hf_token = _get_hf_token()
    if not hf_token:
        return f"⚠️ No HF_TOKEN found in .env.\n\nFallback context:\n{context}"

    try:
        client = InferenceClient(token=hf_token)
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            max_tokens=400,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return (
            f"⚠️ Could not reach HuggingFace API: {e}\n\n"
            f"**Fallback — raw retrieved context:**\n\n{context}"
        )