# 🧠 ChronoRAG — Conversational Memory Intelligence System

> A production-quality RAG system that processes chronological conversation data, detects topic shifts over time, extracts user personas, and answers questions via a ChatGPT-style chatbot.

---

## 🚀 How to Run Locally

### Step 1 — Clone the project
```bash
git clone https://github.com/YOUR_USERNAME/chronaRAG.git
cd chronaRAG
```

### Step 2 — Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your HuggingFace token
Create a `.env` file in the root folder:
```
HF_TOKEN=your_huggingface_token_here
```
Get your free token at: https://huggingface.co/settings/tokens

### Step 5 — Add the dataset
Place your CSV file at:
```
data/conversations.csv
```
The CSV should have one conversation per row (each row = one day of messages).

### Step 6 — Run the app
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

> ⏳ **First load takes ~25 seconds** — it's processing 190,000+ messages, detecting topics, building checkpoints, and indexing everything. After that it's instant (cached).

---

## ☁️ How to Deploy to Cloud (Free — Streamlit Community Cloud)

Anyone with the link can access your hosted app. Here's how:

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/chronaRAG.git
git push -u origin main
```

### Step 2 — Go to Streamlit Cloud
Visit **https://share.streamlit.io** and sign in with GitHub.

### Step 3 — Create a new app
- Click **"New app"**
- Select your GitHub repo: `YOUR_USERNAME/chronaRAG`
- Branch: `main`
- Main file: `app.py`
- Click **"Deploy!"**

### Step 4 — Add your HuggingFace token as a Secret
In Streamlit Cloud dashboard:
- Go to your app → **Settings** → **Secrets**
- Add this:
```toml
HF_TOKEN = "your_huggingface_token_here"
```
- Save and **Reboot app**

### Step 5 — Share your link! 🎉
Your app will be live at:
```
https://YOUR_USERNAME-chronarag-app-XXXXX.streamlit.app
```

---

## 🏗️ Project Architecture

```
chronaRAG/
├── app.py                        # Streamlit UI (Chat / Persona / Topics)
├── requirements.txt              # Dependencies
├── .env                          # HuggingFace token (local only)
├── .gitignore
├── .streamlit/
│   └── secrets.toml              # Token for cloud deployment
├── data/
│   └── conversations.csv         # Input dataset (one day per row)
├── checkpoints/                  # Auto-generated on first run
│   ├── topic_checkpoints.json
│   └── message_checkpoints.json
├── persona/                      # Auto-generated on first run
│   └── persona.json
└── src/
    ├── preprocess.py             # CSV → chronological messages
    ├── topic_detector.py         # Topic change detection (TF-IDF centroid)
    ├── summarizer.py             # Offline extractive summarizer
    ├── checkpoint_manager.py     # Topic + 100-msg checkpoint creation
    ├── persona_extractor.py      # Rule-based persona extraction
    ├── rag_engine.py             # TF-IDF retrieval (3 indexes)
    └── chatbot.py                # LLM response via HuggingFace API
```

---

## 🔍 How Topic Detection Works

> **"Do NOT treat the entire conversation as one topic"** — ✅ Implemented

The system processes messages **strictly in chronological order** and detects semantic topic boundaries using:

1. **TF-IDF Vectorization** — All messages are vectorized in a single `fit_transform` call (fast)
2. **Running Centroid** — The average TF-IDF vector of the last 30 messages represents the "current topic"
3. **Cosine Similarity** — The centroid is compared to the next incoming message
4. **Threshold Split** — When similarity drops below `0.05` AND the topic has `50+` messages → topic boundary

**Output format:**
```
Topic 1  → messages 1  – 51   → "Discussion about books and hobbies"
Topic 2  → messages 52 – 119  → "Talk about jobs and daily routines"
Topic 3  → messages 120 – 171 → "Conversation about food and travel"
```

Topics have **varying, natural sizes** (50–130 messages) based on actual semantic content — not fixed intervals.

---

## 📦 How 100-Message Checkpoints Work

**Independently of topic boundaries**, the system creates a summary every 100 messages:

```
Chunk 1  → messages 1   – 100   → summary
Chunk 2  → messages 101 – 200   → summary
...
Chunk N  → messages X   – X+99  → summary
```

Saved to `checkpoints/message_checkpoints.json`.

---

## 🔎 How Retrieval Works

When a user asks a question, the system queries **3 independent TF-IDF indexes** simultaneously:

| Index | What it searches | Top-K |
|---|---|---|
| **Message Index** | All raw conversation messages | 5 |
| **Topic Index** | Topic checkpoint summaries | 3 |
| **Checkpoint Index** | 100-message chunk summaries | 2 |

All retrieved context is combined into a single grounded prompt sent to the LLM.

---

## 👤 How Persona Extraction Works

The persona is built from **actual conversation evidence only** — no guessing.

| Category | Method |
|---|---|
| **Habits** | Regex patterns: `gym/workout` → Fitness-conscious, `sleep/tired` → Irregular sleeper |
| **Personality Traits** | Regex: `haha/lol` → Humorous, `sad/cry` → Emotional, `stress` → Anxious |
| **Personal Facts** | Regex: `college/exam` → Student, `job/work` → Working/job-seeking |
| **Communication Style** | Count ratios: short message %, emoji frequency, question frequency |

All categories use Python `set` — **impossible to produce duplicate entries**.

---

## 💬 Chatbot Features

- **Chat tab** — ChatGPT-style interface with welcome screen, suggestion cards, and chat bubbles
- **Persona tab** — Visual tags for habits, traits, facts, and communication style
- **Topics tab** — Full chronological topic timeline + 100-message checkpoints side by side

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| NLP / Retrieval | scikit-learn (TF-IDF, cosine similarity) |
| Data Processing | pandas, numpy |
| LLM | Meta-Llama-3-8B-Instruct via HuggingFace Inference API |
| Summarization | Custom extractive summarizer (offline, no API) |

---

## 📋 Dependencies

```
streamlit
scikit-learn
pandas
numpy
huggingface_hub
```

Install with: `pip install -r requirements.txt`
