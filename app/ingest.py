\
from pathlib import Path
from typing import List, Dict
from app.utils import SETTINGS, safe_id
from app.parsing import load_documents

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        if end == n: break
        start = max(0, end - overlap)
        if start >= n: break
    return chunks

def build_chunks_from_folder(folder: Path) -> list[dict]:
    out = []
    for doc in load_documents(folder):
        doc_uid = safe_id(doc["id"])
        pieces = chunk_text(doc["text"], SETTINGS.chunk_size, SETTINGS.chunk_overlap)
        for i, piece in enumerate(pieces):
            out.append({
                "id": f"{doc_uid}#c{i}",
                "text": piece,
                "source": doc["source"],
                "doc_path": doc["id"],
                "doc_uid": doc_uid
            })
    return out
