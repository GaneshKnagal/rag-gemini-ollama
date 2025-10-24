# app/rag.py
from typing import List, Dict, Any, Iterable, Optional
import os
import math

from google import genai
from pinecone import Pinecone, ServerlessSpec  # pip install -U pinecone

import chromadb                                 # pip install chromadb
from chromadb.utils import embedding_functions  # not used directly but good to have

from app.utils import SETTINGS
from app.state import load_ns_state, save_ns_state

# ------------ Provider & Paths ------------
LLM_PROVIDER_DEFAULT = os.getenv("LLM_PROVIDER", "gemini").lower()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")  # 768-d

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")

# ------------ Clients ------------
client = genai.Client(api_key=SETTINGS.gemini_api_key)
pc = Pinecone(api_key=SETTINGS.pinecone_api_key)
_index = None  # Pinecone index handle cache

_chroma = chromadb.PersistentClient(path=CHROMA_DIR)  # local persistent store


# =============== HELPERS ===============
def _normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector (avoid div-by-zero)."""
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


# =============== HEALTH / INDEX (Pinecone) ===============
def pinecone_health() -> dict:
    try:
        pc.list_indexes()
        return {"ok": True, "msg": "Pinecone reachable ✅"}
    except Exception as e:
        return {"ok": False, "msg": f"Pinecone error: {e}"}

def index_exists(name: str) -> bool:
    try:
        return name in [i["name"] for i in pc.list_indexes()]
    except Exception:
        return False

def ensure_index(dim: int = 768) -> None:
    """Create serverless Pinecone index if missing, then cache handle."""
    global _index
    name = SETTINGS.pinecone_index
    if not index_exists(name):
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=SETTINGS.pinecone_cloud,
                region=SETTINGS.pinecone_region,
            ),
        )
    _index = pc.Index(name)

def get_index():
    """Return cached Pinecone index; Gemini embeddings are 768-d."""
    global _index
    if _index is None:
        ensure_index(dim=768)
    return _index


# =============== EMBEDDINGS ===============
# Gemini (cloud)
def _extract_embedding_gemini(resp) -> Optional[List[float]]:
    if resp is None:
        return None
    embs = getattr(resp, "embeddings", None)
    if isinstance(embs, (list, tuple)) and embs:
        vals = getattr(embs[0], "values", None)
        if isinstance(vals, (list, tuple)) and vals:
            return list(vals)
    # fallbacks for older shapes:
    if isinstance(resp, dict) and "embedding" in resp:
        return list(resp["embedding"])
    emb = getattr(resp, "embedding", None)
    if isinstance(emb, (list, tuple)):
        return list(emb)
    out = getattr(resp, "output", None)
    if out is not None:
        emb = getattr(out, "embedding", None)
        if isinstance(emb, (list, tuple)):
            return list(emb)
    return None

def embed_texts_gemini(texts: List[str]) -> List[List[float]]:
    vecs: List[List[float]] = []
    for t in texts:
        try:
            resp = client.models.embed_content(model="text-embedding-004", contents=t)
            v = _extract_embedding_gemini(resp)
            if v:
                vecs.append(v)
        except Exception:
            pass
    return vecs

# Ollama (local) — normalize to stabilize cosine distance
def embed_texts_ollama(texts: List[str]) -> List[List[float]]:
    try:
        from ollama import Client
    except Exception:
        return []
    oc = Client(host=OLLAMA_HOST)
    vecs: List[List[float]] = []
    for t in texts:
        try:
            r = oc.embeddings(model=OLLAMA_EMBED_MODEL, prompt=t)
            v = r.get("embedding")
            if isinstance(v, list):
                vecs.append(_normalize(v))
        except Exception:
            pass
    return vecs

def embed_texts(texts: List[str], provider: str) -> List[List[float]]:
    provider = provider.lower()
    if provider == "ollama":
        return embed_texts_ollama(texts)
    return embed_texts_gemini(texts)


# =============== VECTOR STORES ===============
# ----- Pinecone -----
def upsert_chunks_pinecone(chunks: List[Dict[str, Any]], namespace: Optional[str]=None, provider: str="gemini"):
    index = get_index()
    vecs = embed_texts([c["text"] for c in chunks], provider="gemini")
    if not vecs or len(vecs) != len(chunks):
        raise RuntimeError(f"Embedding mismatch (Gemini): expected {len(chunks)}, got {len(vecs)}.")
    items = []
    for c, v in zip(chunks, vecs):
        items.append({
            "id": c["id"],
            "values": v,
            "metadata": {
                "source": c["source"],
                "chunk": c["text"][:4000],
                "doc_uid": c.get("doc_uid"),
                "doc_path": c.get("doc_path"),
            },
        })
    B = 100
    for i in range(0, len(items), B):
        index.upsert(items[i:i+B], namespace=namespace)

def query_pinecone(q: str, k: int, namespace: Optional[str]=None) -> List[Dict[str, Any]]:
    index = get_index()
    qv = embed_texts([q], provider="gemini")
    if not qv:
        return []
    res = index.query(vector=qv[0], top_k=k, include_metadata=True, namespace=namespace)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    hits = []
    for m in matches:
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
        md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
        hits.append({"score": float(score), "source": md.get("source","unknown"), "chunk": md.get("chunk","")})
    return hits

def delete_by_ids_pinecone(ids: List[str], namespace: Optional[str]=None):
    get_index().delete(ids=ids, namespace=namespace)

def delete_namespace_pinecone(namespace: str):
    get_index().delete(delete_all=True, namespace=namespace)

# ----- Chroma (local) -----
def _chroma_collection(namespace: str):
    """
    One collection per namespace, **forced to cosine** space to avoid huge L2 distances.
    Recreate the collection (drop namespace) if you previously made it without this metadata.
    """
    name = f"rag_{namespace}"
    try:
        return _chroma.get_collection(name)
    except Exception:
        # Ensure cosine space so distances ∈ [0, 2]
        return _chroma.create_collection(name=name, metadata={"hnsw:space": "cosine"})

def upsert_chunks_chroma(chunks: List[Dict[str, Any]], namespace: str, provider: str="ollama"):
    col = _chroma_collection(namespace)
    vecs = embed_texts([c["text"] for c in chunks], provider="ollama")
    if not vecs or len(vecs) != len(chunks):
        raise RuntimeError(f"Embedding mismatch (Ollama): expected {len(chunks)}, got {len(vecs)}")
    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "chunk": c["text"][:4000], "doc_uid": c.get("doc_uid"), "doc_path": c.get("doc_path")} for c in chunks]
    col.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)

def query_chroma(q: str, k: int, namespace: str, provider: str="ollama") -> List[Dict[str, Any]]:
    col = _chroma_collection(namespace)
    qv = embed_texts([q], provider="ollama")
    if not qv:
        return []
    # Normalize query to match stored vectors
    qv = [_normalize(qv[0])]
    res = col.query(query_embeddings=qv, n_results=k, include=["metadatas", "distances"])
    hits: List[Dict[str, Any]] = []
    if not res or not res.get("metadatas"):
        return hits
    dists = res.get("distances", [[]])[0] if res.get("distances") else []
    metas = res["metadatas"][0]
    for dist, md in zip(dists, metas):
        if dist is None:
            sim = 0.0
        else:
            d = float(dist)
            # For cosine space: distance d ∈ [0, 2]; similarity ≈ 1 - d ∈ [-1, 1]
            sim = 1.0 - d
            # Fallback if something looks like L2 (very large distances)
            if d > 2.5 or sim < -1.0:
                sim = 1.0 / (1.0 + d)
        # Clamp for UX/hallucination gate
        sim = max(0.0, min(1.0, sim))
        hits.append({"score": sim, "source": md.get("source","unknown"), "chunk": md.get("chunk","")})
    return hits

def delete_by_docuid_chroma(namespace: str, doc_uid: str):
    col = _chroma_collection(namespace)
    col.delete(where={"doc_uid": doc_uid})

def drop_namespace_chroma(namespace: str):
    try:
        _chroma.delete_collection(f"rag_{namespace}")
    except Exception:
        pass


# =============== PUBLIC API (provider-aware) ===============
def upsert_chunks(chunks: List[Dict[str, Any]], namespace: Optional[str]=None, provider: str="gemini"):
    if provider.lower() == "ollama":
        if not namespace:
            raise ValueError("Chroma mode requires a namespace string.")
        return upsert_chunks_chroma(chunks, namespace, provider="ollama")
    return upsert_chunks_pinecone(chunks, namespace=namespace, provider="gemini")

def similarity_search(q: str, k: int, namespace: Optional[str]=None, provider: str="gemini") -> List[Dict[str, Any]]:
    if provider.lower() == "ollama":
        if not namespace:
            return []
        return query_chroma(q, k, namespace, provider="ollama")
    return query_pinecone(q, k, namespace,)

def delete_by_ids(ids: List[str], namespace: Optional[str]=None, provider: str="gemini"):
    if provider.lower() == "ollama":
        return
    return delete_by_ids_pinecone(ids, namespace)

def delete_namespace(namespace: str, provider: str="gemini"):
    if provider.lower() == "ollama":
        return drop_namespace_chroma(namespace)
    return delete_namespace_pinecone(namespace)

def delete_document_by_uid(namespace: str, doc_uid: str, provider: str="gemini"):
    if provider.lower() == "ollama":
        return delete_by_docuid_chroma(namespace, doc_uid)
    # pinecone path uses stored ids (handled by delta_ingest)


# =============== PROMPT & GENERATION (Gemini/Ollama) ===============
def build_prompt(q: str, hits: List[Dict[str, Any]], history: Optional[list]=None) -> str:
    ctx = "\n\n".join([f"[S{i}] Source: {h['source']}\n{h['chunk']}" for i, h in enumerate(hits, 1)])
    system = (
        "You are a helpful assistant that MUST answer ONLY from the provided sources.\n"
        "Cite sources inline like [S1], [S2]. If the answer is not in the sources, say:\n"
        "\"I couldn’t find this information in the provided documents.\""
    )
    hist_block = ""
    if history:
        pairs = []
        for turn in history[-6:]:
            role = turn.get("role", "user"); text = turn.get("content", "")
            pairs.append(f"{role.upper()}: {text}")
        hist_block = "\n\nCHAT HISTORY (recent):\n" + "\n".join(pairs)
    return f"{system}\n\nSOURCES:\n{ctx}\n{hist_block}\n\nQuestion: {q}\nAnswer:"

def stream_answer_gemini(prompt: str) -> Iterable[str]:
    # Try streaming; if SDK doesn't support, fall back to non-stream
    try:
        stream = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, stream=True)
        for event in stream:
            text = getattr(event, "text", None)
            if not text:
                candidates = getattr(event, "candidates", [])
                if candidates:
                    parts = getattr(candidates[0].content, "parts", [])
                    if parts and hasattr(parts[0], "text"):
                        text = parts[0].text
            if text:
                yield text
        return
    except Exception:
        pass
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = getattr(resp, "text", None)
        if not text:
            candidates = getattr(resp, "candidates", [])
            if candidates:
                parts = getattr(candidates[0].content, "parts", [])
                if parts and hasattr(parts[0], "text"):
                    text = parts[0].text
        yield text or "I couldn’t generate a response."
    except Exception:
        yield "I couldn’t generate a response."

def stream_answer_ollama(prompt: str) -> Iterable[str]:
    try:
        from ollama import Client
    except Exception:
        yield "Ollama client not installed. Run: pip install -U ollama"
        return
    oc = Client(host=OLLAMA_HOST)
    options = {"keep_alive": "5m", "num_ctx": 2048, "num_predict": 256, "temperature": 0.2}
    for chunk in oc.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], stream=True, options=options):
        m = chunk.get("message")
        if m and "content" in m:
            yield m["content"]
        elif "response" in chunk:
            yield chunk["response"]

def stream_answer(prompt: str, provider: Optional[str]=None) -> Iterable[str]:
    p = (provider or LLM_PROVIDER_DEFAULT).lower()
    if p == "ollama":
        yield from stream_answer_ollama(prompt)
    else:
        yield from stream_answer_gemini(prompt)


# =============== DELTA INGEST & DELETES (provider-aware) ===============
def delta_ingest(chunks: List[Dict[str, Any]], namespace: str, provider: str="gemini") -> dict:
    """
    Only upsert changed/new files' chunks and delete vectors for removed files.
    Tracked in state/<namespace>.json as {uid: {path, chunk_ids}}.
    Works for both Pinecone (gemini) and Chroma (ollama).
    """
    state = load_ns_state(namespace)
    tracked = state.get("docs", {})  # uid -> {path, chunk_ids}

    # Build snapshot from incoming chunks
    snapshot: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        uid = c["doc_uid"]; path = c["doc_path"]
        snapshot.setdefault(uid, {"path": path, "chunk_ids": []})
        snapshot[uid]["chunk_ids"].append(c["id"])

    # Deleted files
    deleted_doc_uids = [uid for uid in tracked.keys() if uid not in snapshot]

    # New/changed files
    new_or_changed_uids = []
    for uid, info in snapshot.items():
        if uid not in tracked:
            new_or_changed_uids.append(uid)
        else:
            prev = set(tracked[uid].get("chunk_ids", []))
            cur = set(info["chunk_ids"])
            if prev != cur:
                new_or_changed_uids.append(uid)

    changed_chunks = [c for c in chunks if c["doc_uid"] in new_or_changed_uids]

    # Delete removed docs' vectors
    if deleted_doc_uids:
        if provider.lower() == "ollama":
            for uid in deleted_doc_uids:
                delete_by_docuid_chroma(namespace, uid)
        else:
            ids = []
            for uid in deleted_doc_uids:
                ids.extend(tracked.get(uid, {}).get("chunk_ids", []))
            if ids:
                delete_by_ids_pinecone(ids, namespace=namespace)
        for uid in deleted_doc_uids:
            tracked.pop(uid, None)

    # Upsert changed/new
    if changed_chunks:
        if provider.lower() == "ollama":
            upsert_chunks_chroma(changed_chunks, namespace, provider="ollama")
        else:
            upsert_chunks_pinecone(changed_chunks, namespace=namespace, provider="gemini")
        for uid in new_or_changed_uids:
            tracked[uid] = {"path": snapshot[uid]["path"], "chunk_ids": snapshot[uid]["chunk_ids"]}

    state["docs"] = tracked
    save_ns_state(namespace, state)

    return {
        "upserted_chunks": len(changed_chunks),
        "deleted_docs": len(deleted_doc_uids),
        "tracked_docs": len(tracked),
    }

def delete_document_by_uid(namespace: str, doc_uid: str, provider: str="gemini"):
    if provider.lower() == "ollama":
        delete_by_docuid_chroma(namespace, doc_uid)
    else:
        state = load_ns_state(namespace); tracked = state.get("docs", {})
        ids = tracked.get(doc_uid, {}).get("chunk_ids", [])
        if ids:
            delete_by_ids_pinecone(ids, namespace=namespace)
        tracked.pop(doc_uid, None)
        state["docs"] = tracked
        save_ns_state(namespace, state)

def drop_namespace(namespace: str, provider: str="gemini"):
    if provider.lower() == "ollama":
        drop_namespace_chroma(namespace)
        save_ns_state(namespace, {"docs": {}})
    else:
        delete_namespace_pinecone(namespace)
        save_ns_state(namespace, {"docs": {}})
