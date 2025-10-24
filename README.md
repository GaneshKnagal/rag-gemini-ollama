# 📚 RAG — Gemini / Ollama + Pinecone/Chroma (Streamlit)

Dual-mode Retrieval-Augmented Generation (RAG) app:

- **Gemini mode** → Google AI `text-embedding-004` + **Pinecone** vector DB + (Gemini 2.5 Flash) generation
- **Ollama mode** → Local embeddings (e.g., `nomic-embed-text`) + **ChromaDB** vector DB + (Mistral) generation

## ✨ Features
- Upload **PDF/Markdown** in the UI or index everything in a **docs/** folder
- **Citations** for each answer
- **Streaming** responses (Ollama path; Gemini falls back if SDK streaming is unavailable)
- **Guardrails**: profanity filter + similarity threshold hallucination gate
- **Delta ingest**: only new/changed files re-embedded
- **Namespaces**: keep multiple datasets separate
- **Latency** display (ms)

> **Supported files**: `.pdf`, `.md`, `.markdown`

---

## 🧱 Project Structure

```
.
├─ app/
│  ├─ __init__.py
│  ├─ ingest.py          # loads/cleans/splits docs into chunks
│  ├─ rag.py             # dual-mode RAG core (Gemini/Pinecone or Ollama/Chroma)
│  ├─ state.py           # per-namespace tracking for delta ingest
│  ├─ utils.py           # settings loader, timers, helpers
│  ├─ filters.py         # profanity + hallucination gate
├─ docs/                 # optional: sample docs; also used by UI uploads
├─ chroma_store/         # local Chroma DB (ignored by git)
├─ state/                # ingest tracking (ignored by git)
├─ .streamlit/
│  └─ config.toml
├─ .gitignore
├─ requirements.txt
├─ streamlit_app.py      # main UI
├─ .env.example          # template (no secrets)
└─ README.md
```

---

## 🔑 Environment

Copy `.env.example` → `.env` (don’t commit `.env`). Fill:

```env
# Default provider used on app start (can switch in UI)
LLM_PROVIDER=gemini    # gemini | ollama

# Google AI (Gemini)
GEMINI_API_KEY=PUT_YOUR_KEY_HERE

# Pinecone (for Gemini mode)
PINECONE_API_KEY=PUT_YOUR_KEY_HERE
PINECONE_INDEX=rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Chunking (better recall with smaller overlap)
CHUNK_SIZE=600
CHUNK_OVERLAP=150

# Retrieval defaults
TOP_K=5
MIN_SIM_THRESHOLD=0.55
PROFANITY_BLOCK=1

# Ollama (local mode)
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=mistral
OLLAMA_EMBED_MODEL=nomic-embed-text

# Chroma local store path
CHROMA_DIR=./chroma_store
```

---

## 🛠️ Install & Run (Local)

```bash
# 1) Create env and install deps
python -m venv .venv
# or: conda create -n rag_llm python=3.10
pip install -r requirements.txt

# 2) (Optional) for local mode
#    install & run Ollama, then pull models:
#    https://ollama.com/download
ollama serve
ollama pull mistral
ollama pull nomic-embed-text

# 3) Run the app
streamlit run streamlit_app.py
```

---

## 🖥️ Using the App

1. Choose **LLM provider** in sidebar:
   - **Gemini** → embeddings + Pinecone (cloud)
   - **Ollama** → local embeddings + Chroma (local)
2. Upload files (PDF/MD) or place them in the **Documents folder** (default `docs/`)
3. Click **Save uploads to folder** (for uploaded files)
4. Click **Scan & Ingest (delta)** to index the folder
5. Ask questions → answers cite sources. Adjust **Top-K** and **Min similarity** as needed.

**Sidebar tips**
- **Top-K**: 3–7 (start with 5)
- **Min similarity**: 0.50–0.60 for small corpora
- Namespaces keep datasets separate (e.g., `productA`, `productB`).

---

## ☁️ Deploy on Streamlit Community Cloud

> **Note:** Ollama (local) doesn’t run on Streamlit Cloud. Deploy **Gemini + Pinecone** mode.

1. Push repo to GitHub.
2. Streamlit Cloud → **New app** → select your repo/branch → `streamlit_app.py`.
3. Add **Secrets** (App → Settings → Secrets), e.g.:
   ```toml
   GEMINI_API_KEY = "YOUR_KEY"
   PINECONE_API_KEY = "YOUR_KEY"
   PINECONE_INDEX = "rag-index"
   PINECONE_CLOUD = "aws"
   PINECONE_REGION = "us-east-1"
   LLM_PROVIDER = "gemini"
   TOP_K = "5"
   MIN_SIM_THRESHOLD = "0.55"
   CHUNK_SIZE = "600"
   CHUNK_OVERLAP = "150"
   ```
4. Deploy. Use the **Upload** widget or commit docs to the repo.

---

## 🔎 Troubleshooting

- **Gemini embeddings fail** → ensure `google-genai>=0.2.0` and `GEMINI_API_KEY` is set.
- **Pinecone empty** → create index via the top status panel (768, cosine) and check region/cloud.
- **Ollama embeddings mismatch** → run `ollama serve` and `ollama pull nomic-embed-text`.
- **Negative scores in Ollama mode** → collection must be cosine; we force `metadata={'hnsw:space':'cosine'}` and normalize vectors. Drop namespace and re-ingest.
- **Low recall** → set `CHUNK_SIZE=600`, `CHUNK_OVERLAP=150`, `TOP_K=5`, `MIN_SIM_THRESHOLD=0.55`.

---

## 🔐 Security / Secrets

- Don’t commit real API keys. Use `.env` locally and **Secrets** on Streamlit Cloud.
- `.gitignore` excludes `.env`, `state/`, and `chroma_store/`.

---

## 📜 License

MIT © 2025 Your Name
