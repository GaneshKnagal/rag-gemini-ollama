import os
import streamlit as st
from pathlib import Path
from app.utils import SETTINGS, now_ms
from app.ingest import build_chunks_from_folder
from app.rag import (
    upsert_chunks, similarity_search, build_prompt, stream_answer,
    pinecone_health, index_exists, ensure_index,
    delta_ingest, delete_document_by_uid, drop_namespace
)
from app.state import load_ns_state
from app.filters import contains_profanity, hallucination_gate

st.set_page_config(page_title="RAG — Gemini / Ollama + Pinecone/Chroma", page_icon="📚", layout="wide")
st.title("📚 RAG — Gemini / Ollama + Pinecone/Chroma")

# ---------- Helpers ----------
def save_uploaded_files(uploaded_files, target_dir: Path) -> list[Path]:
    """Save uploaded files to disk and return the saved paths."""
    saved = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        out_path = target_dir / uf.name
        with open(out_path, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(out_path)
    return saved

# ---------- Provider first (we need it early) ----------
with st.sidebar:
    st.header("Settings")
    default_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    provider_options = ["gemini", "ollama"]
    try:
        default_index = provider_options.index(default_provider)
    except ValueError:
        default_index = 0
    provider = st.selectbox(
        "LLM provider",
        provider_options,
        index=default_index,
        help="Gemini: cloud gen + Gemini embeddings + Pinecone.  Ollama: local gen + local embeddings + Chroma."
    )

# ---------- Health / Index ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Vector Backend Health")
    if provider == "gemini":
        health = pinecone_health()
        st.success(health["msg"]) if health["ok"] else st.error(health["msg"])
    else:
        st.info("Chroma (local) — no remote health check required ✅")

with col2:
    st.subheader("Index Status")
    if provider == "gemini":
        name = SETTINGS.pinecone_index
        exists = index_exists(name)
        if exists:
            st.success(f"Pinecone index '{name}' exists ✅")
        else:
            st.warning(f"Index '{name}' not found ❌")
            if st.button("Create index now (768, cosine)"):
                ensure_index(dim=768)
                st.success(f"Index '{name}' created ✅")
    else:
        st.success("Chroma collections are created on demand ✅")

st.divider()

# ---------- Sidebar (main controls) ----------
with st.sidebar:
    top_k = st.number_input("Top-K", 1, 20, SETTINGS.top_k)
    min_sim = st.slider("Min similarity (hallucination gate)", 0.0, 1.0, SETTINGS.min_sim_threshold, 0.01)
    namespace = st.text_input("Namespace", value="default")
    use_history = st.checkbox("Use chat history in prompt", value=True)
    show_debug = st.checkbox("Show retrieval scores (debug)", value=False)
    st.caption("Namespaces separate datasets/versions.")

    st.divider()
    st.header("Ingestion")

    # Folder path (retained)
    docs_dir = st.text_input("Documents folder", "docs")

    # Upload new files into that folder
    st.caption("Upload files (saved into the folder above) — supported: **.pdf**, **.md**, **.markdown**")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "md", "markdown"],
        accept_multiple_files=True,
        help="Drop PDFs or Markdown files here. They will be saved under the folder shown above."
    )

    c1, c2 = st.columns(2)
    with c1:
        save_btn = st.button("Save uploads to folder")
    with c2:
        ingest_btn = st.button("Scan & Ingest (delta)")

    folder = Path(docs_dir)

    # Save uploads
    if save_btn:
        if not folder.exists():
            st.error("Folder does not exist. Please create it or change the path.")
        elif not uploaded_files:
            st.warning("No files selected.")
        else:
            saved_paths = save_uploaded_files(uploaded_files, folder)
            st.success(f"Saved {len(saved_paths)} file(s) to `{folder}`.")
            st.caption("Now click “Scan & Ingest (delta)” to index them.")

    # Ingest from whatever is currently in the folder
    if ingest_btn:
        if not folder.exists():
            st.error("Folder does not exist")
        else:
            try:
                with st.status(f"Scanning & delta ingest… (mode: {provider})", expanded=False):
                    chunks = build_chunks_from_folder(folder)
                    stats = delta_ingest(chunks, namespace=namespace, provider=provider)
                    st.write(stats)
                    st.success("Delta ingestion complete ✅")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                if provider == "ollama":
                    st.caption("If provider = 'ollama', ensure 'ollama serve' is running and an embedding model "
                               "is pulled (e.g., 'ollama pull nomic-embed-text').")

    st.divider()
    st.header("Manage Data")
    ns_state = load_ns_state(namespace)
    docs_map = ns_state.get("docs", {})
    all_docs = [(uid, info.get("path","")) for uid, info in docs_map.items()]
    if all_docs:
        labels = [f"{uid} — {path}" for uid, path in all_docs]
        choice = st.selectbox("Select a document to delete", labels, index=0)
        if st.button("Delete selected document"):
            sel_uid = all_docs[labels.index(choice)][0]
            delete_document_by_uid(namespace, sel_uid, provider=provider)
            st.success(f"Deleted document {sel_uid} in namespace '{namespace}'")
    else:
        st.caption("No tracked documents in this namespace yet.")

    if st.button("⚠️ Drop entire namespace (irreversible)"):
        drop_namespace(namespace, provider=provider)
        st.success(f"Dropped namespace '{namespace}' (vectors deleted).")

# ---------- Ask ----------
st.subheader("Ask a question")
q = st.text_input("Your question", placeholder="Type here…")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
ns_hist = st.session_state.chat_history.setdefault(namespace, [])

if st.button("Ask") and q:
    t0 = now_ms()
    if SETTINGS.profanity_block and contains_profanity(q):
        st.error("Please avoid profanity.")
    else:
        hits = similarity_search(q, k=top_k, namespace=namespace, provider=provider)
        max_score = max([h["score"] for h in hits], default=0.0)
        allowed, msg = hallucination_gate(max_score, min_sim)

        if show_debug:
            st.caption(f"Top score: {max_score:.3f}")
            st.json(hits)

        ns_hist.append({"role": "user", "content": q})

        if not allowed:
            st.info(msg)
            ns_hist.append({"role": "assistant", "content": msg})
        else:
            prompt = build_prompt(q, hits, history=ns_hist if use_history else None)
            st.write(f"**Answer (streaming via {provider}):**")
            ph = st.empty(); acc = ""

            # Stream from chosen provider (Gemini/Ollama)
            ttfb_ms = None
            for token in stream_answer(prompt, provider=provider):
                if ttfb_ms is None:
                    ttfb_ms = now_ms() - t0
                acc += token
                # Update less frequently to reduce UI overhead
                if len(acc) % 20 == 0:
                    ph.markdown(acc)
            ph.markdown(acc)

            if hits:
                st.write("**Sources:**")
                cols = st.columns(min(len(hits), 4))
                for i, h in enumerate(hits):
                    with cols[i % len(cols)]:
                        st.caption(f"[S{i+1}] {h['source']} (score={h['score']:.2f})")
            ns_hist.append({"role": "assistant", "content": acc})

    dt = now_ms() - t0
    st.info(f"Latency: {dt} ms ({dt/1000:.2f} s)")

# ---------- History ----------
if ns_hist:
    st.divider()
    st.subheader(f"Chat history (namespace: {namespace})")
    for turn in ns_hist[-8:]:
        who = "🧑‍💻 You" if turn["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{who}:** {turn['content']}")
