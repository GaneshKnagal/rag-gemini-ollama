\
from pathlib import Path
import fitz  # PyMuPDF
import markdown as md
from html import unescape
import re

def read_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    html = md.markdown(text)
    return unescape(re.sub(r"<[^>]+>", " ", html))

def read_pdf(path: Path) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)

def load_documents(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file():
            suf = p.suffix.lower()
            if suf in (".md", ".markdown"):
                yield {"id": str(p), "text": read_markdown(p), "source": p.name}
            elif suf == ".pdf":
                yield {"id": str(p), "text": read_pdf(p), "source": p.name}
