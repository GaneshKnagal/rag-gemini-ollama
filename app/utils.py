\
import os, time, hashlib
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "rag-index")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1200))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    top_k: int = int(os.getenv("TOP_K", 5))
    min_sim_threshold: float = float(os.getenv("MIN_SIM_THRESHOLD", 0.75))
    profanity_block: bool = bool(int(os.getenv("PROFANITY_BLOCK", 1)))

SETTINGS = Settings()

def now_ms(): return int(time.time()*1000)

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def md5_file(path: str) -> str:
    with open(path, "rb") as f: return md5_bytes(f.read())

def safe_id(s: str) -> str:
    import hashlib as _h
    return _h.sha1(s.encode("utf-8")).hexdigest()[:16]
