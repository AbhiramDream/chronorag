"""
app.py — ChronoRAG Streamlit Application
Clean ChatGPT-style chat UI with Persona and Topics tabs.
"""

import json
import os
import streamlit as st

from src.preprocess import load_conversations
from src.topic_detector import detect_topics
from src.checkpoint_manager import create_topic_checkpoints, create_message_checkpoints
from src.persona_extractor import extract_persona
from src.rag_engine import initialize_rag
from src.chatbot import generate_response

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChronoRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: ChatGPT-style dark UI ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #212121;
    color: #ececec;
}
[data-testid="stHeader"] { background: transparent; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #2f2f2f;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #aaa;
    border-radius: 8px;
    font-weight: 500;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #404040 !important;
    color: #fff !important;
}

/* Chat messages */
.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.user-bubble {
    background: #2f2f2f;
    color: #ececec;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.6;
}
.bot-message {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    gap: 12px;
    align-items: flex-start;
}
.bot-avatar {
    background: linear-gradient(135deg, #7f5af0, #2cb67d);
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.bot-bubble {
    background: #2f2f2f;
    color: #ececec;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.6;
}

/* Welcome screen */
.welcome-wrap {
    text-align: center;
    padding: 80px 20px 40px;
}
.welcome-logo {
    font-size: 3rem;
    margin-bottom: 12px;
}
.welcome-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ececec;
    margin-bottom: 8px;
}
.welcome-sub {
    color: #aaa;
    font-size: 1rem;
    margin-bottom: 40px;
}
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 600px;
    margin: 0 auto;
}
.suggestion-card {
    background: #2f2f2f;
    border: 1px solid #404040;
    border-radius: 12px;
    padding: 14px 16px;
    cursor: pointer;
    text-align: left;
    color: #ccc;
    font-size: 14px;
    line-height: 1.4;
}
.suggestion-card:hover { border-color: #7f5af0; color: #fff; }

/* Stat cards */
.stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
.stat-card {
    flex: 1;
    background: #2f2f2f;
    border: 1px solid #3f3f3f;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #7f5af0; }
.stat-lbl { font-size: 0.78rem; color: #888; margin-top: 4px; }

/* Persona */
.persona-section { background: #2f2f2f; border-radius: 12px; padding: 20px; margin-bottom: 14px; }
.persona-section h4 { color: #7f5af0; margin: 0 0 12px 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
.persona-tag {
    display: inline-block;
    background: #3a3a3a;
    border: 1px solid #505050;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 13px;
    color: #ddd;
    margin: 3px;
}

/* Topic card */
.topic-card {
    background: #2f2f2f;
    border-left: 3px solid #7f5af0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 14px;
}
.topic-meta { color: #888; font-size: 12px; margin-bottom: 4px; }
.topic-summary { color: #ccc; }

/* Input box override */
[data-testid="stChatInput"] {
    background: #2f2f2f !important;
    border-radius: 16px !important;
    border: 1px solid #404040 !important;
}
</style>
""", unsafe_allow_html=True)


# ── One-time pipeline (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Processing conversations…")
def setup_pipeline():
    messages = load_conversations("data/conversations.csv")
    topics   = detect_topics(messages)
    t_cps    = create_topic_checkpoints(topics)
    m_cps    = create_message_checkpoints(messages)
    persona  = extract_persona(messages)
    initialize_rag(messages)
    return persona, topics, t_cps, m_cps, messages

persona, topics, topic_cps, msg_cps, messages = setup_pipeline()


# ── Session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 16px 0 8px 0; border-bottom: 1px solid #333; margin-bottom: 16px;'>
  <span style='font-size:1.5rem; font-weight:700; background:linear-gradient(90deg,#7f5af0,#2cb67d);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🧠 ChronoRAG</span>
  <span style='color:#666; font-size:0.85rem; margin-left:12px;'>Conversational Memory Intelligence</span>
</div>
""", unsafe_allow_html=True)

# Stats bar
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in [
    (c1, f"{len(messages):,}", "Messages"),
    (c2, f"{len(topics):,}", "Topics Detected"),
    (c3, f"{len(topic_cps):,}", "Topic Checkpoints"),
    (c4, f"{len(msg_cps):,}", "100-Msg Checkpoints"),
]:
    col.markdown(
        f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_chat, tab_persona, tab_topics = st.tabs(["💬  Chat", "👤  Persona", "📋  Topics"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — CHAT (ChatGPT-style)
# ════════════════════════════════════════════════════════════════
with tab_chat:

    # ── Welcome screen (shown only when no messages) ──────────────
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-wrap">
          <div class="welcome-logo">🧠</div>
          <div class="welcome-title">ChronoRAG</div>
          <div class="welcome-sub">Ask anything about the user's conversations, habits, or personality</div>
        </div>
        """, unsafe_allow_html=True)

        # Suggestion buttons (2×2 grid)
        suggestions = [
            ("🧑 What kind of person is this user?",  "What kind of person is this user?"),
            ("🛌 What are their habits?",              "What are their habits?"),
            ("💬 How do they communicate?",            "How does this user communicate?"),
            ("😄 Are they humorous or serious?",       "Is this user humorous or serious?"),
        ]
        col_a, col_b = st.columns(2)
        for i, (label, query) in enumerate(suggestions):
            col = col_a if i % 2 == 0 else col_b
            if col.button(label, key=f"sug_{i}", use_container_width=True):
                st.session_state.prefill = query
                st.rerun()

    # ── Chat history ──────────────────────────────────────────────
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-message"><div class="user-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                content = msg["content"].replace("\n", "<br>")
                st.markdown(
                    f'''<div class="bot-message">
                          <div class="bot-avatar">🧠</div>
                          <div class="bot-bubble">{content}</div>
                        </div>''',
                    unsafe_allow_html=True
                )

        if st.button("🗑️ New chat", key="clear"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Chat input (always visible at bottom) ──────────────────────
    # Handle suggestion prefill
    if st.session_state.prefill:
        query = st.session_state.prefill
        st.session_state.prefill = ""
    else:
        query = st.chat_input("Message ChronoRAG…")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner(""):
            answer = generate_response(query)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


# ════════════════════════════════════════════════════════════════
# TAB 2 — PERSONA
# ════════════════════════════════════════════════════════════════
with tab_persona:
    st.markdown("### 👤 Extracted User Persona")
    st.caption("Built purely from conversation signals — no guessing or hallucination.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Habits
        habits = persona.get("habits", [])
        tags = "".join(f'<span class="persona-tag">🛌 {h}</span>' for h in habits) or "<span style='color:#666'>None detected</span>"
        st.markdown(f'<div class="persona-section"><h4>Habits</h4>{tags}</div>', unsafe_allow_html=True)

        # Personal facts
        facts = persona.get("personal_facts", [])
        tags = "".join(f'<span class="persona-tag">ℹ️ {f}</span>' for f in facts) or "<span style='color:#666'>None detected</span>"
        st.markdown(f'<div class="persona-section"><h4>Personal Facts</h4>{tags}</div>', unsafe_allow_html=True)

    with col_r:
        # Traits
        traits = persona.get("personality_traits", [])
        tags = "".join(f'<span class="persona-tag">🎭 {t}</span>' for t in traits) or "<span style='color:#666'>None detected</span>"
        st.markdown(f'<div class="persona-section"><h4>Personality Traits</h4>{tags}</div>', unsafe_allow_html=True)

        # Communication style
        style = persona.get("communication_style", {})
        items = "".join(
            f'<span class="persona-tag">💬 {k.replace("_"," ").title()}: {v}</span>'
            for k, v in style.items()
        ) or "<span style='color:#666'>None detected</span>"
        st.markdown(f'<div class="persona-section"><h4>Communication Style</h4>{items}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄 View raw persona.json"):
        st.json(persona)


# ════════════════════════════════════════════════════════════════
# TAB 3 — TOPICS
# ════════════════════════════════════════════════════════════════
with tab_topics:
    st.markdown("### 📋 Topic Timeline")
    st.caption(
        f"**{len(topic_cps)}** topics detected chronologically · "
        f"**{len(msg_cps)}** 100-message checkpoints · "
        f"**{len(messages):,}** total messages"
    )

    col_t, col_m = st.columns([3, 2])

    with col_t:
        st.markdown("#### 🏷️ Topic Checkpoints")
        st.caption("Each entry = one detected topic segment")
        # Show first 100 to keep UI responsive
        shown = topic_cps[:100]
        for cp in shown:
            st.markdown(
                f'''<div class="topic-card">
                  <div class="topic-meta">Topic {cp["topic_id"]} &nbsp;·&nbsp; Messages {cp["start_msg"]} → {cp["end_msg"]} &nbsp;·&nbsp; {cp.get("message_count","?")} messages</div>
                  <div class="topic-summary">{cp["summary"] or "—"}</div>
                </div>''',
                unsafe_allow_html=True
            )
        if len(topic_cps) > 100:
            st.caption(f"Showing first 100 of {len(topic_cps)} topics.")

    with col_m:
        st.markdown("#### 📦 100-Message Checkpoints")
        st.caption("Independent of topics — every 100 messages")
        for i, cp in enumerate(msg_cps, 1):
            st.markdown(
                f'''<div class="topic-card" style="border-left-color:#2cb67d;">
                  <div class="topic-meta">Chunk {i} &nbsp;·&nbsp; Messages {cp["start_msg"]} → {cp["end_msg"]}</div>
                  <div class="topic-summary">{cp["summary"] or "—"}</div>
                </div>''',
                unsafe_allow_html=True
            )
