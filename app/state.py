\
import json
from pathlib import Path
from typing import Dict, Any

STATE_ROOT = Path(__file__).resolve().parent.parent / "state"

def _ns_path(namespace: str) -> Path:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return STATE_ROOT / f"{namespace}.json"

def load_ns_state(namespace: str) -> Dict[str, Any]:
    p = _ns_path(namespace)
    if not p.exists(): return {"docs": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"docs": {}}

def save_ns_state(namespace: str, data: Dict[str, Any]) -> None:
    p = _ns_path(namespace)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
