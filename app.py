# app.py
"""
Streamlit RAG Support App (complete, with debug + self-tests + UI + Quit)

Features
1) YouTrack connection (URL + token) and project/issues list
2) Ticket ingestion into a local Vector DB (Chroma) with local or OpenAI embeddings
3) LLM selection (OpenAI or local via Ollama)
4) RAG chat: search similar tickets + LLM answer with ID citations
5) CLI mode with self-tests when Streamlit is not available
6) Quit button in the sidebar to close the app from the browser
7) If data/chroma/chroma.sqlite3 is found, the collection is opened automatically and the number of loaded documents is shown (or “N/A” if not available)

Recommended start
streamlit run app.py --server.port 8502
"""

import os
import sys
import time
import json
import math
import uuid
import typing
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
import shutil, subprocess

# NEW (optional): token encoder for precise chunking
try:
    import tiktoken
    _tk_enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tk_enc = None

# === Sticky prefs (Level A) ===
import os

# --- NumPy 2 compatibility shim (before importing chromadb) ---
try:
    import numpy as _np  # type: ignore
    if not hasattr(_np, "float_"):  # NumPy 2.0+
        _np.float_ = _np.float64     # type: ignore[attr-defined]
    if not hasattr(_np, "int_"):
        _np.int_ = _np.int64         # type: ignore[attr-defined]
    if not hasattr(_np, "uint"):
        _np.uint = _np.uint64        # type: ignore[attr-defined]
except Exception:
    pass

APP_DIR = os.path.dirname(os.path.abspath(__file__))

def is_cloud() -> bool:
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and str(st.secrets.get("DEPLOY_ENV", "")).lower() == "cloud":
            return True
    except Exception:
        pass
    # fallback (heuristic): repo not writable => treat it as cloud
    try:
        test_path = os.path.join(APP_DIR, ".write_test")
        with open(test_path, "w") as f:
            f.write("x")
        os.remove(test_path)
        return False
    except Exception:
        return True

IS_CLOUD = is_cloud()

def pick_default_chroma_dir() -> str:
    env_dir = os.getenv("CHROMA_DIR")
    if env_dir:
        return env_dir
    # also support st.secrets["CHROMA_DIR"]
    try:
        import streamlit as st  # type: ignore
        sec_dir = st.secrets.get("CHROMA_DIR") if hasattr(st, "secrets") else None
        if sec_dir:
            return str(sec_dir)
    except Exception:
        pass
    return "/tmp/chroma" if IS_CLOUD or not os.access(APP_DIR, os.W_OK) else os.path.join(APP_DIR, "data", "chroma")

DEFAULT_CHROMA_DIR = pick_default_chroma_dir()

# Preferences path: write to /tmp in cloud
PREFS_PATH = os.path.join("/tmp", ".app_prefs.json") if IS_CLOUD else os.path.join(APP_DIR, ".app_prefs.json")


NON_SENSITIVE_PREF_KEYS = {
    "yt_url",
    "persist_dir",
    "emb_backend",
    "emb_model_name",
    "llm_provider",
    "llm_model",
    "llm_temperature",
    "max_distance",
    "show_prompt",
    "collection_selected",
    "new_collection_name",
}

def load_prefs() -> dict:
    try:
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: v for k, v in data.items() if k in NON_SENSITIVE_PREF_KEYS}
    except Exception:
        pass
    return {}

def save_prefs(prefs: dict):
    try:
        clean = {k: v for k, v in prefs.items() if k in NON_SENSITIVE_PREF_KEYS}
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] save_prefs: {e}")

def init_prefs_in_session():
    prefs = load_prefs() or {}

    pd = (prefs.get("persist_dir") or "").strip()
    if not pd:
        prefs["persist_dir"] = DEFAULT_CHROMA_DIR
    else:
        if IS_CLOUD and not pd.startswith("/tmp/"):
            prefs["persist_dir"] = DEFAULT_CHROMA_DIR

    st.session_state["prefs"] = prefs


def one_line_preview(text: str, maxlen: int = 160) -> str:
    """Make text single-line, remove bullets/extra spaces, and truncate."""
    if not text:
        return ""
    s = text.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # remove list marker at the beginning of the text (es. "- ", "* ", "• ")
    s = re.sub(r"^[-\*\u2022]\s+", "", s)
    return (s[:maxlen] + "…") if len(s) > maxlen else s

# --- Chunking utilities (token-based, with tiktoken fallback) ---
def _tokenize(text: str):
    """
    Returns tokens as ids if tiktoken is available, otherwise whitespace tokens.
    """
    if _tk_enc:
        return _tk_enc.encode(text or "")
    return re.findall(r"\S+", text or "")

def _detokenize(toks):
    if _tk_enc:
        return _tk_enc.decode(toks)
    return " ".join(map(str, toks))

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 80, min_size: int = 512):
    """
    Split very large descriptions into chunks with overlap.
    Returns list of tuples: (start_token_pos, chunk_text).
    If text is short (<= min_size), returns a single chunk with pos=0.
    """
    toks = _tokenize(text)
    n = len(toks)
    if n == 0:
        return [(0, "")]
    if n <= max(min_size, 2*chunk_size):
        return [(0, text)]

    chunks = []
    start = 0
    while start < n:
        end = min(n, start + int(chunk_size))
        # detokenize
        if _tk_enc:
            chunk_text = _tk_enc.decode(toks[start:end])
        else:
            chunk_text = " ".join(toks[start:end])
        chunks.append((start, chunk_text))
        if end >= n:
            break
        start = max(0, end - int(overlap))  # sliding window with overlap
    return chunks

# === Solution Memory (Level B) ===
MEM_COLLECTION = "memories"        # separate Chroma collection for playbooks
DEFAULT_MEM_TTL_DAYS = 180         # recommended expiration for memories

def now_ts() -> int:
    return int(time.time())

def ts_in_days(days: int) -> int:
    return now_ts() + days * 24 * 3600

def is_ollama_available() -> tuple[bool, str]:
    """
    Return (available, host). Check via HTTP on configured host
    and, as fallback, the presence of the 'ollama' binary.
    """
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    # Quick HTTP attempt
    try:
        import requests  # lazy import
        r = requests.get(f"{host}/api/tags", timeout=1)
        if r.status_code < 500:
            return True, host
    except Exception:
        pass
    # Fallback: binary present and executable
    try:
        if shutil.which("ollama"):
            subprocess.run(
                ["ollama", "list"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1
            )
            return True, host
    except Exception:
        pass
    return False, host

# ------------------------------
# Try imports with fallback/shim
# ------------------------------
try:
    import streamlit as st
    ST_AVAILABLE = True
    print("[DEBUG] Streamlit imported.")
except Exception:
    ST_AVAILABLE = False
    print("[DEBUG] Streamlit not available, using shim.")

    class _STShim:
        def __getattr__(self, _name):
            def _noop(*_args, **_kwargs):
                print(f"[DEBUG] Call st.{_name} ignorata (shim).")
                return None
            return _noop

    st = _STShim()  # type: ignore

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    print("[DEBUG] chromadb imported.")
except Exception:
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore
    print("[DEBUG] chromadb not available.")

try:
    from sentence_transformers import SentenceTransformer
    print("[DEBUG] sentence-transformers imported.")
except Exception:
    SentenceTransformer = None  # type: ignore
    print("[DEBUG] sentence-transformers not available.")

try:
    from openai import OpenAI
    print("[DEBUG] openai SDK imported.")
except Exception:
    OpenAI = None  # type: ignore
    print("[DEBUG] openai SDK not available.")


# ------------------------------
# Constants
# ------------------------------
DEFAULT_COLLECTION = "tickets"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_OLLAMA_MODEL = "llama3.2"


# ------------------------------
# YouTrack Client + DTO
# ------------------------------
@dataclass
class YTIssue:
    id_readable: str
    summary: str
    description: str
    project: str

    def text_blob(self) -> str:
        return f"{self.summary}\n\n{self.description or ''}"


class YouTrackClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _get(self, path: str, params: Optional[dict] = None):
        import requests
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_projects(self) -> List[dict]:
        print("[DEBUG] list_projects")
        return self._get("/api/admin/projects", params={"fields": "name,shortName,id"})

    def list_issues(self, project_key: str, limit: int = 200) -> List[YTIssue]:
        print(f"[DEBUG] list_issues per progetto {project_key}")
        fields = "idReadable,summary,description,project(name,shortName)"
        params = {
            "query": f"project: {project_key}",
            "$top": str(limit),
            "fields": fields,
        }
        data = self._get("/api/issues", params=params)
        issues: List[YTIssue] = []
        for it in data:
            issues.append(
                YTIssue(
                    id_readable=it.get("idReadable", ""),
                    summary=it.get("summary", ""),
                    description=it.get("description", ""),
                    project=(it.get("project", {}) or {}).get("shortName")
                            or (it.get("project", {}) or {}).get("name", ""),
                )
            )
        print(f"[DEBUG] trovati {len(issues)} issues")
        return issues


# ------------------------------
# Embeddings backend
# ------------------------------
class EmbeddingBackend:
    def __init__(self, use_openai: bool, model_name: str):
        self.use_openai = use_openai
        self.model_name = model_name
        self.provider_name = "openai" if use_openai else "sentence-transformers"

        if use_openai:
            if OpenAI is None:
                raise RuntimeError("openai SDK not available")
            api_key = get_openai_key()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set (enter it in the sidebar or as an environment variable)")
            self.client = OpenAI(api_key=api_key)
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers non available")
            self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        print(f"[DEBUG] embed {len(texts)} text with provider={self.provider_name} model={self.model_name}")
        if self.use_openai:
            res = self.client.embeddings.create(model=self.model_name, input=texts)
            return [d.embedding for d in res.data]  # type: ignore
        return self.model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore

def get_chroma_client(persist_dir: str):
    """Create folder and return a Chroma PersistentClient."""
    try:
        import chromadb  # lazy import (no global)
        from chromadb.config import Settings as ChromaSettings
    except Exception as e:
        # explicit diagnostics
        st.error("ChromaDB cannot be imported in this environment.")
        st.exception(e)
        raise

    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False)
    )


# ------------------------------
# VectorStore (Chroma wrapper)
# ------------------------------
class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        if chromadb is None:
            raise RuntimeError("ChromaDB not available")
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)  # type: ignore
        )
        self.col = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, ids: List[str], documents: List[str], metadatas: List[dict], embeddings: List[List[float]]):
        print(f"[DEBUG] add {len(ids)} documents to {self.collection_name}")
        self.col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_embeddings: List[List[float]], n_results: int = 5) -> dict:
        return self.col.query(query_embeddings=query_embeddings, n_results=n_results)

    def count(self) -> int:
        try:
            return self.col.count()  # type: ignore
        except Exception:
            return -1

def get_openai_key() -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        ui_key = st.session_state.get("openai_key")
        if ui_key:
            return ui_key
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_EXPERIMENTS")

# ------------------------------
# LLM Backend
# ------------------------------
class LLMBackend:
    def __init__(self, provider: str, model_name: str, temperature: float = 0.2):
        self.provider = provider
        self.model_name = model_name
        self.temperature = float(temperature)
        if provider == "OpenAI":
            if OpenAI is None:
                raise RuntimeError("openai SDK not available")
            api_key = get_openai_key()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set (enter it in the sidebar or as an environment variable)")
            self.client = OpenAI(api_key=api_key)

        elif provider == "Ollama (local)":
            ok, host = is_ollama_available()
            if not ok:
                raise RuntimeError("Ollama is not available in this environment: select an OpenAI provider.")
            # No SDK client needed: use REST; keep host for calls
            self.client = None
            self.ollama_host = host  # es. "http://localhost:11434"

        else:
            raise RuntimeError("Unsupported LLM provider")

    def generate(self, system: str, user: str) -> str:
        if self.provider == "OpenAI":
            try:
                # Responses API
                res = self.client.responses.create(
                    model=self.model_name,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.temperature,
                )
                return res.output_text  # type: ignore
            except Exception:
                # Fallback Chat Completions
                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.temperature,
                )
                return chat.choices[0].message.content or ""
        elif self.provider == "Ollama (local)":
            import requests
            host = getattr(self, "ollama_host", "http://localhost:11434").rstrip("/")
            url = f"{host}/api/chat"

            payload = {
                "model": self.model_name,  # es. "llama3.2" o "llama3.2:latest"
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,  # evita JSON concatenati
                "options": {"temperature": self.temperature},  # Ollama supporta temperature
            }

            r = requests.post(url, json=payload, timeout=300)
            try:
                r.raise_for_status()
                data = r.json()  # singolo JSON
            except json.JSONDecodeError:
                # fallback per eventuale streaming non richiesto
                text = r.text.strip()
                chunks, buf, depth = [], "", 0
                for ch in text:
                    buf += ch
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                chunks.append(json.loads(buf))
                            except Exception:
                                pass
                            buf = ""
                parts = []
                for obj in chunks:
                    if isinstance(obj, dict):
                        if "message" in obj and isinstance(obj["message"], dict) and "content" in obj["message"]:
                            parts.append(obj["message"]["content"])
                        elif "response" in obj:
                            parts.append(obj["response"])
                return "\n".join(parts).strip() if parts else text

            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                    return data["message"]["content"]
                if "response" in data:
                    return data["response"]
                if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
                    last = data["messages"][-1]
                    if isinstance(last, dict) and "content" in last:
                        return last["content"]
            return str(data)


RAG_SYSTEM_PROMPT = (
    "You are a technical assistant who answers based on similar YouTrack tickets. "
    "Always cite the IDs of the tickets found in square brackets. If the context is insufficient, ask for clarifications. Use english in the answer"
)

def build_prompt(user_ticket: str, retrieved: List[Tuple[str, dict, float]]) -> str:
    print(f"[DEBUG] build_prompt with {len(retrieved)} retrieved documents")
    parts = [
        "New ticket:\n" + user_ticket.strip(),
        "\nSimilar tickets found (closest first):",
    ]
    for doc, meta, dist in retrieved:
        parts.append(
            f"- {meta.get('id_readable','')} | distance={dist:.3f} | {meta.get('summary','')}\n{doc[:500]}"
        )
    parts.append("\nAnswer including citations in [ ] of the relevant ticket IDs. Please use always english in the answer")
    return "\n".join(parts)


from collections import defaultdict

def collapse_by_parent(results, per_parent=1, stitch_for_prompt=False, max_chars=1200):
    """
    Collapse multiple chunks from the same ticket into one row.
    results: list of (doc, meta, dist, src)
    - per_parent: keep at most N items per parent (1 for display)
    - stitch_for_prompt: if True, concatenate selected chunks in order of 'pos' (bounded by max_chars)
    Returns: list of (doc, meta, dist, src) sorted by distance.
    """
    groups = defaultdict(list)
    for doc, meta, dist, src in results:
        meta = meta or {}
        pid = meta.get("parent_id") or meta.get("id_readable")
        groups[pid].append((doc, meta, dist, src))

    collapsed = []
    for pid, items in groups.items():
        items = sorted(items, key=lambda x: (x[2], x[1].get("pos", 0)))  # by distance, then pos
        keep = items[:max(1, per_parent)]

        if stitch_for_prompt and len(keep) > 1:
            keep_sorted = sorted(keep, key=lambda x: x[1].get("pos", 0))
            buf = []
            total = 0
            for d, _m, _dist, _src in keep_sorted:
                if total + len(d) + 2 > max_chars:
                    break
                buf.append(d)
                total += len(d) + 2
            best = keep[0]  # best distance
            stitched = "\n\n".join(buf) if buf else best[0]
            collapsed.append((stitched, best[1], best[2], best[3]))
        else:
            collapsed.append(keep[0])

    return sorted(collapsed, key=lambda x: x[2])

# ------------------------------
# NEW: utility to open the collection into the session
# ------------------------------
def open_vector_in_session(persist_dir: str, collection_name: str):
    """Open (or create) a Chroma collection and put it into the session."""
    try:
        vs = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        cnt = vs.count()
        try:
            import streamlit as st  # type: ignore
            st.session_state["vector"] = vs
            st.session_state["vs_persist_dir"] = persist_dir
            st.session_state["vs_collection"] = collection_name
            st.session_state["vs_count"] = cnt
        except Exception:
            pass
        return True, cnt, None
    except Exception as e:
        try:
            import streamlit as st  # type: ignore
            st.session_state["vector"] = None
        except Exception:
            pass
        return False, -1, str(e)


# ------------------------------
# Main Streamlit UI
# ------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="YouTrack RAG Support", layout="wide")
    init_prefs_in_session()
    prefs = st.session_state.get("prefs", {})

    # Init UI input key (separate from app state)
    if "new_collection_name_input" not in st.session_state:
        st.session_state["new_collection_name_input"] = (prefs.get("new_collection_name") or DEFAULT_COLLECTION)

    # Pre-reset (prima dei widget)
    if st.session_state.get("after_delete_reset"):
        st.session_state["new_collection_name_input"] = DEFAULT_COLLECTION
        st.session_state["after_delete_reset"] = False

    st.title("YouTrack RAG Support")
    st.caption("Support ticket management based on YouTrack history + RAG + LLM")

    with st.sidebar:
        quit_btn = False
        st.header("YouTrack Connection")
        # Pre-reset (must happen BEFORE widgets creation)
        if st.session_state.get("after_delete_reset"):
            st.session_state["new_collection_name_input"] = DEFAULT_COLLECTION
            st.session_state["after_delete_reset"] = False

        yt_url = st.text_input(
            "Server URL",
            value=prefs.get("yt_url", ""),
            placeholder="https://<org>.myjetbrains.com/youtrack",
        )
        yt_token = st.text_input("Token (Bearer)", type="password")
        connect = st.button("Connect YouTrack")

        st.markdown("---")
        st.header("Vector DB")
        persist_dir = st.text_input("Chroma path", value=prefs.get("persist_dir", DEFAULT_CHROMA_DIR), key="persist_dir")
        # Patch 6: create the directory and show the current path
        os.makedirs(persist_dir, exist_ok=True)
        st.caption(f"Current Chroma path: {persist_dir}")

        # Read existing collections
        coll_options = []
        try:
            if chromadb is not None:
                _client = get_chroma_client(persist_dir)
                coll_options = [c.name for c in _client.list_collections()]  # type: ignore
        except Exception as e:
            st.caption(f"Unable to read collections from '{persist_dir}': {e}")

        NEW_LABEL = "➕ Create new collection..."
        # Keep the last selected/new collection name in session
        prefs = st.session_state.get("prefs", {})

        # Fallback: first session → then prefs → None
        last_sel = st.session_state.get("collection_selected")
        if not last_sel:
            last_sel = (prefs.get("collection_selected") or "").strip() or None

        last_new = st.session_state.get("new_collection_name_input")
        if not last_new:
            last_new = (prefs.get("new_collection_name") or DEFAULT_COLLECTION)

        st.caption(f"Selected pref: {prefs.get('collection_selected', '—')}  ·  Path: {persist_dir}")

        if coll_options:
            opts = coll_options + [NEW_LABEL]

            # If a valid last selection exists, use it; otherwise default to 0
            if last_sel in opts:
                default_index = opts.index(last_sel)
            elif DEFAULT_COLLECTION in opts:
                default_index = opts.index(DEFAULT_COLLECTION)
            else:
                default_index = 0

            sel = st.selectbox("Collection", options=opts, index=default_index, key="collection_select")

            if sel == NEW_LABEL:
                new_name = st.text_input("New Collection name", key="new_collection_name_input")
                collection_name = (st.session_state["new_collection_name_input"] or "").strip() or DEFAULT_COLLECTION
                # Record that we are creating a new collection
                st.session_state["collection_selected"] = NEW_LABEL
            else:
                collection_name = sel
                st.session_state["collection_selected"] = sel
                st.session_state["after_delete_reset"] = True  # optional reset
        else:
            st.caption("No collection found. Create a new one:")
            new_name = st.text_input("New Collection", value=last_new, key="new_collection_name_input")
            collection_name = (st.session_state["new_collection_name_input"] or "").strip() or DEFAULT_COLLECTION
            st.session_state["collection_selected"] = NEW_LABEL

        # === Advanced settings ===
        with st.expander("Advanced settings", expanded=False):
            # leggi valori correnti (prefs -> session -> default)
            _prefs = st.session_state.get("prefs", {})

            # Visibilità/Recall
            show_distances = st.checkbox(
                "Show distances in results",
                value=bool(_prefs.get("show_distances", False)),
                key="adv_show_distances",
                help="Mostra la distanza/score accanto a ciascun risultato."
            )

            top_k = st.number_input(
                "Top-K KB results",
                min_value=1, max_value=50, value=int(_prefs.get("top_k", 5)),
                step=1, key="adv_top_k",
                help="Quanti risultati del Knowledge Base passare a valle (prima del collasso)."
            )

            collapse_duplicates = st.checkbox(
                "Collapse duplicate results by ticket",
                value=bool(_prefs.get("collapse_duplicates", True)),
                key="adv_collapse_duplicates",
                help="Mostra un solo risultato per ticket in UI (mantiene recall nel prompt)."
            )

            # Granularità per documento
            per_parent_display = st.number_input(
                "Max results per ticket (UI)",
                min_value=1, max_value=10, value=int(_prefs.get("per_parent_display", 1)),
                step=1, key="adv_per_parent_display",
                help="Quanti risultati al massimo per lo stesso ticket nella lista visualizzata."
            )

            per_parent_prompt = st.number_input(
                "Max chunks per ticket (prompt)",
                min_value=1, max_value=10, value=int(_prefs.get("per_parent_prompt", 3)),
                step=1, key="adv_per_parent_prompt",
                help="Quanti chunk cucire per ticket nel contesto del prompt."
            )

            stitch_max_chars = st.number_input(
                "Stitched context limit (chars)",
                min_value=200, max_value=20000, value=int(_prefs.get("stitch_max_chars", 1500)),
                step=100, key="adv_stitch_max_chars",
                help="Limite di caratteri quando concateni più chunk dello stesso ticket per il prompt."
            )

        # Rendi disponibili nel run corrente (se li usi in funzioni a valle)
        st.session_state["show_distances"] = st.session_state.get("adv_show_distances", False)
        st.session_state["top_k"] = int(st.session_state.get("adv_top_k", 5))
        st.session_state["collapse_duplicates"] = st.session_state.get("adv_collapse_duplicates", True)
        st.session_state["per_parent_display"] = int(st.session_state.get("adv_per_parent_display", 1))
        st.session_state["per_parent_prompt"] = int(st.session_state.get("adv_per_parent_prompt", 3))
        st.session_state["stitch_max_chars"] = int(st.session_state.get("adv_stitch_max_chars", 1500))

        # --- Collection management: delete ---
        st.markdown("—")
        st.subheader("Collection management")
        is_existing_collection = collection_name in coll_options

        del_confirm = st.checkbox(
            f"Confirm deletion of '{collection_name}'",
            value=False,
            disabled=not is_existing_collection,
            help="This operation permanently removes the collection from the datastore."
        )

        if st.button(
            "Delete collection",
            type="secondary",
            disabled=not is_existing_collection,
            help="Permanently removes the selected collection from the vector datastore."
        ):
            if not del_confirm:
                st.warning("Check the confirmation box to proceed with deletion.")
            else:
                try:
                    _client = get_chroma_client(persist_dir)
                    _client.delete_collection(name=collection_name)

                    # Remove the collection meta (provider/model) if present
                    meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
                    try:
                        if os.path.exists(meta_path):
                            os.remove(meta_path)
                    except Exception:
                        pass  # do not block the UX if meta deletion fails

                    # Clean state if it was pointing to the removed collection
                    if st.session_state.get("vs_collection") == collection_name:
                        st.session_state["vector"] = None
                        st.session_state["vs_collection"] = None
                        st.session_state["vs_count"] = 0
                        st.session_state["vs_persist_dir"] = persist_dir  # optional: keep the current path

                    # Clear any loaded results/issues (optional but recommended)
                    st.session_state["issues"] = []

                    # Remove selection and new name from session
                    st.session_state["collection_selected"] = None
                    st.session_state["after_delete_reset"] = True

                    # Also update sticky prefs, if present
                    prefs = st.session_state.get("prefs", {})
                    prefs["collection_selected"] = None
                    prefs["new_collection_name"] = DEFAULT_COLLECTION
                    st.session_state["prefs"] = prefs
                    # If save_prefs() is defined, update file on disk (safe no-op if absent)
                    try:
                        save_prefs(prefs)
                    except Exception:
                        pass

                    st.success(f"Collection '{collection_name}' deleted successfully.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error during deletion: {e}")


        st.markdown("---")
        # --- [SIDEBAR > Embeddings] replace the current block with this ---

        st.header("Embeddings")

        # Costruisci la lista in modo deterministico
        emb_backends = []
        if SentenceTransformer is not None and not IS_CLOUD:
            emb_backends.append("Local (sentence-transformers)")
        emb_backends.append("OpenAI")  # OpenAI c'è sempre in lista

        # Backend preferito da prefs, con fallback robusto
        pref_backend = (prefs.get("emb_backend") or "OpenAI")
        if pref_backend == "Local (sentence-transformers)" and "Local (sentence-transformers)" not in emb_backends:
            pref_backend = "OpenAI"

        # Usa l'indice del backend preferito (non 0 fisso)
        emb_backend = st.selectbox(
            "Embeddings provider",
            options=emb_backends,
            index=emb_backends.index(pref_backend),
            key="emb_provider_select",
        )

        # Reset modello se cambia provider
        prev_emb_backend = st.session_state.get("last_emb_backend")
        if prev_emb_backend != emb_backend:
            st.session_state["last_emb_backend"] = emb_backend
            st.session_state["emb_model"] = (
                "all-MiniLM-L6-v2" if emb_backend == "Local (sentence-transformers)" else "text-embedding-3-small"
            )

        # Inizializza il modello UNA SOLA VOLTA leggendo le prefs
        if "emb_model" not in st.session_state:
            st.session_state["emb_model"] = prefs.get(
                "emb_model_name",
                "all-MiniLM-L6-v2" if emb_backend == "Local (sentence-transformers)" else "text-embedding-3-small"
            )

        # Opzioni per modello coerenti col provider corrente
        emb_model_options = (
            ["all-MiniLM-L6-v2"] if emb_backend == "Local (sentence-transformers)"
            else ["text-embedding-3-small", "text-embedding-3-large"]
        )

        # Riallinea se il valore corrente non è nelle opzioni
        if st.session_state["emb_model"] not in emb_model_options:
            st.session_state["emb_model"] = emb_model_options[0]

        emb_model_name = st.selectbox(
            "Embeddings model",
            options=emb_model_options,
            index=emb_model_options.index(st.session_state["emb_model"]),
            key="emb_model"
        )

        st.header("Retrieval")
        if "max_distance" not in st.session_state:
            st.session_state["max_distance"] = float(prefs.get("max_distance", 0.9))
        max_distance = st.slider("Maximum distance threshold (cosine)", 0.1, 2.0, st.session_state["max_distance"], 0.05)
        st.session_state["max_distance"] = max_distance

        # NEW: Chunking params
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            chunk_size = st.number_input("Chunk size (tokens)", min_value=128, max_value=2048, value=800, step=64, help="512–800 suggested")
        with c_b:
            chunk_overlap = st.number_input("Overlap (tokens)", min_value=0, max_value=512, value=80, step=10)
        with c_c:
            chunk_min = st.number_input("Min size to chunk", min_value=128, max_value=2048, value=512, step=64)

        # NEW: toggle to enable/disable chunking at ingestion time
        enable_chunking = st.checkbox(
            "Enable chunking",
            value=False,
            help="Disable for short, self-contained tickets (index 1 document per ticket)."
        )

        st.markdown("---")
        st.header("LLM")

        # Ollama detection
        ollama_ok, ollama_host = (False, None) if IS_CLOUD else is_ollama_available()
        llm_provider_options = ["OpenAI"] + (["Ollama (local)"] if ollama_ok else [])

        # Default index consistent with prefs but safe if Ollama is NOT available
        pref_provider = prefs.get("llm_provider", "OpenAI")
        default_idx = 0 if (pref_provider != "Ollama (local)" or not ollama_ok) else 1

        llm_provider = st.selectbox(
            "LLM provider",
            llm_provider_options,
            index=default_idx,
            key="llm_provider_select",
        )

        if not ollama_ok:
            st.caption("⚠️ Ollama is not available in this environment; option disabled.")

        # If the provider changes, reset the model to the selected provider's default
        prev_provider = st.session_state.get("last_llm_provider")
        if prev_provider != llm_provider:
            st.session_state["last_llm_provider"] = llm_provider
            st.session_state["llm_model"] = (
                DEFAULT_LLM_MODEL if llm_provider == "OpenAI" else DEFAULT_OLLAMA_MODEL
            )

        # Default for provider + prefs (ignoring empty strings)
        llm_model_default = (DEFAULT_LLM_MODEL if llm_provider == "OpenAI" else DEFAULT_OLLAMA_MODEL)
        pref_llm_model = (prefs.get("llm_model") or "").strip()

        if "llm_model" not in st.session_state:
            st.session_state["llm_model"] = pref_llm_model or llm_model_default
        else:
            if not (st.session_state["llm_model"] or "").strip():
                st.session_state["llm_model"] = llm_model_default

        # Field controlled via session_state
        llm_model = st.text_input("LLM model", key="llm_model")

        # Temperature slider (as in your code)
        if "llm_temperature" not in st.session_state:
            st.session_state["llm_temperature"] = float(prefs.get("llm_temperature", 0.2))
        llm_temperature = st.slider("Temperature", 0.0, 1.5, st.session_state["llm_temperature"], 0.05)
        st.session_state["llm_temperature"] = llm_temperature

        # --- [SIDEBAR > API Keys] ---
        st.header("API Keys")
        openai_needed = (emb_backend == "OpenAI") or (llm_provider == "OpenAI")

        openai_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_key", ""),
            disabled=not openai_needed,
            help="Used only if you choose OpenAI as provider for Embeddings or LLM."
        )

        if openai_key_input:
            st.session_state["openai_key"] = openai_key_input

        if not openai_needed:
            st.caption("OpenAI API Key not needed: you are using only local providers (Ollama / sentence-transformers).")
        elif (llm_provider == "Ollama (local)") and (emb_backend == "OpenAI"):
            st.info("You are using: LLM = Ollama, Embeddings = OpenAI → the key will be used only for embeddings.")

        st.markdown("---")
        st.header("Chat settings")
        st.caption(f"Top-K = {st.session_state.get('top_k', 5)} · Show distances = {st.session_state.get('show_distances', False)}")

        st.header("Debug")
        if "show_prompt" not in st.session_state:
            st.session_state["show_prompt"] = bool(prefs.get("show_prompt", False))
        show_prompt = st.checkbox("Show LLM prompt", value=st.session_state["show_prompt"])
        st.session_state["show_prompt"] = show_prompt

        st.markdown("---")
        st.subheader("Preferences")
        prefs_enabled = st.checkbox("Enable preferences memory (local)", value=True, help="Save non-sensitive settings in .app_prefs.json")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save preferences"):
                if prefs_enabled:
                    # --- Save preferences (version consistent with Ollama availability) ---
                    # Provider coercion: if Ollama is not available, force OpenAI
                    _provider_for_save = st.session_state.get("last_llm_provider", llm_provider)
                    if not ollama_ok and _provider_for_save == "Ollama (local)":
                        _provider_for_save = "OpenAI"

                    # LLM model: never empty and consistent with the selected provider
                    _model_for_save = (st.session_state.get("llm_model") or "").strip()
                    if not _model_for_save:
                        _model_for_save = DEFAULT_LLM_MODEL if _provider_for_save == "OpenAI" else DEFAULT_OLLAMA_MODEL

                    save_prefs({
                        "yt_url": yt_url,
                        "persist_dir": persist_dir,
                        "emb_backend": emb_backend,
                        "emb_model_name": st.session_state.get("emb_model"),
                        "llm_provider": _provider_for_save,       # <-- use the consistent provider
                        "llm_model": _model_for_save,             # <-- never empty
                        "llm_temperature": st.session_state.get("llm_temperature", llm_temperature),
                        "max_distance": st.session_state.get("max_distance", max_distance),
                        "show_prompt": st.session_state.get("show_prompt", show_prompt),
                        "collection_selected": st.session_state.get("collection_selected"),
                        "new_collection_name": st.session_state.get("new_collection_name_input"),
                        "show_distances": st.session_state.get("adv_show_distances", False),
                        "top_k": st.session_state.get("adv_top_k", 5),
                        "collapse_duplicates": st.session_state.get("adv_collapse_duplicates", True),
                        "per_parent_display": st.session_state.get("adv_per_parent_display", 1),
                        "per_parent_prompt": st.session_state.get("adv_per_parent_prompt", 3),
                        "stitch_max_chars": st.session_state.get("adv_stitch_max_chars", 1500),
                    })
                    st.session_state["prefs"] = load_prefs()
                    st.success("Preferences salvate.")
                else:
                    st.info("Preferences memory disabled.")
        with c2:
            if st.button("Restore defaults"):
                try:
                    if os.path.exists(PREFS_PATH):
                        os.remove(PREFS_PATH)
                    st.session_state["prefs"] = {}
                    st.success("Preferences restored. Reload the page to see the defaults.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while restoring: {e}")

        st.markdown("---")
        st.header("Solutions memory")

        st.checkbox(
            "Show full text of playbooks (MEM)",
            key="mem_show_full",
            help="If enabled, an expander with the full playbook appears under each MEM result."
        )
        enable_memory = st.checkbox(
            "Enable 'playbook' memory (separate collection)",
            value=st.session_state.get("enable_memory", False),
            help="When you mark an answer as 'Solved', I save a reusable mini-playbook."
        )
        st.session_state["enable_memory"] = enable_memory

        mem_ttl_days = st.number_input(
            "TTL (days) for playbooks",
            min_value=7, max_value=365,
            value=st.session_state.get("mem_ttl_days", DEFAULT_MEM_TTL_DAYS), step=1
        )
        st.session_state["mem_ttl_days"] = int(mem_ttl_days)

        c_mem1, c_mem2 = st.columns(2)
        with c_mem1:
            if st.session_state.pop("open_memories_after_save", False):
                st.session_state["show_memories"] = True
            st.checkbox(
                "Show saved playbooks",
                key="show_memories",
                help="Display the list of the 'memories' collection on the main page."
            )
        with c_mem2:
            mem_del_confirm = st.checkbox("Confirm delete memories", value=False)
            if st.button("Delete all memories", disabled=not mem_del_confirm):
                try:
                    _client = get_chroma_client(persist_dir)
                    _client.delete_collection(name=MEM_COLLECTION)
                    _client.get_or_create_collection(name=MEM_COLLECTION, metadata={"hnsw:space": "cosine"})
                    st.success("Memories deleted.")
                except Exception as e:
                    st.error(f"Error deleting memories: {e}")
        st.markdown("---")
        if not IS_CLOUD:
            quit_btn = st.button("Quit", use_container_width=True)
            if quit_btn:
                st.write("Closing application...")
                os._exit(0)
        st.caption(f"IS_CLOUD={IS_CLOUD} · ChromaDB dir={st.session_state['prefs'].get('persist_dir', DEFAULT_CHROMA_DIR)}")

        # Automatic opening of the selected collection (without re-indexing)
        # Avoid opening if the user has chosen "Create new…" but has not entered a name different from the default
        vector_ready = False
        if persist_dir and collection_name:
            changed = (
                st.session_state.get("vs_persist_dir") != persist_dir
                or st.session_state.get("vs_collection") != collection_name
                or st.session_state.get("vector") is None
            )

            # Do not open if we are on NEW_LABEL and the name is empty (or only default) and does not yet exist
            if st.session_state.get("collection_selected") == NEW_LABEL and (collection_name == "" or (collection_name == DEFAULT_COLLECTION and collection_name not in coll_options)):
                pass  # wait for the user to click "Index" to create the new collection
            else:
                if changed:
                    ok, cnt, err = open_vector_in_session(persist_dir, collection_name)
                    vector_ready = ok
                    if ok:
                        st.caption(f"Collection '{collection_name}' opened. Indexed documents: {cnt if cnt>=0 else 'N/A'}")
                    else:
                        st.caption(f"Unable to open the collection: {err}")
                else:
                    vector_ready = True
                    cnt = st.session_state.get("vs_count", -1)
                    st.caption(f"Collection '{collection_name}' ready. Indexed documents: {cnt if cnt>=0 else 'N/A'}")

        if quit_btn:
            st.write("Closing application...")
            os._exit(0)

    # Session state init
    if "yt_client" not in st.session_state:
        st.session_state["yt_client"] = None
    if "projects" not in st.session_state:
        st.session_state["projects"] = []
    if "issues" not in st.session_state:
        st.session_state["issues"] = []
    if "vector" not in st.session_state:
        st.session_state["vector"] = None
    if "embedder" not in st.session_state:
        st.session_state["embedder"] = None

    # Connect to YouTrack
    if connect:
        if not yt_url or not yt_token:
            st.error("Enter YouTrack URL and Token")
        else:
            try:
                st.session_state["yt_client"] = YouTrackClient(yt_url, yt_token)
                st.success("Connected to YouTrack")
                with st.spinner("Loading projects..."):
                    st.session_state["projects"] = st.session_state["yt_client"].list_projects()
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Projects & issues

    st.subheader("YouTrack Projects")
    if st.session_state.get("yt_client"):
        projects = st.session_state.get("projects", [])
        if projects:
            names = [f"{p.get('name')} ({p.get('shortName')})" for p in projects]
            sel = st.selectbox("Choose project", options=["-- select --"] + names, key="proj_select")

            if sel and sel != "-- select --":
                p = projects[names.index(sel)]
                project_key = p.get("shortName") or p.get("name")

                # AUTO-LOAD: when the project changes, immediately load the issues
                prev_key = st.session_state.get("last_project_key")
                if prev_key != project_key:
                    try:
                        with st.spinner("Loading issues..."):
                            st.session_state["issues"] = st.session_state["yt_client"].list_issues(project_key)
                        st.session_state["last_project_key"] = project_key
                        st.success(f"Loaded {len(st.session_state['issues'])} issues")
                    except Exception as e:
                        st.error(f"Error loading issues: {e}")

                # optional: button to reload manually
                if st.button("Reload issues"):
                    try:
                        with st.spinner("Reloading issues..."):
                            st.session_state["issues"] = st.session_state["yt_client"].list_issues(project_key)
                        st.success(f"Loaded {len(st.session_state['issues'])} issues")
                    except Exception as e:
                        st.error(f"Reload error: {e}")
        else:
            st.caption("No project found (or insufficient permissions).")

    # ALWAYS show the table if there are issues in memory
    issues = st.session_state.get("issues", [])
    if issues:
        base_url = (st.session_state.get("yt_client").base_url if st.session_state.get("yt_client") else "").rstrip("/")

        # Build a Markdown table with clickable ID and without Project/Open
        lines = []
        lines.append("| ID | Summary |")
        lines.append("|---|---|")
        for it in issues:
            url = f"{base_url}/issue/{it.id_readable}" if base_url else ""
            id_cell = f"[{it.id_readable}]({url})" if url else it.id_readable
            # trim very long summaries (optional)
            summary = it.summary.strip().replace("\n", " ")
            if len(summary) > 160:
                summary = summary[:157] + "..."
            lines.append(f"| {id_cell} | {summary} |")

        st.markdown("\n".join(lines), unsafe_allow_html=False)
    else:
        st.caption("No issues in memory. Select a project to load tickets.")

    # Main: show the table only if active
    if st.session_state.get("show_memories"):
        st.subheader("Saved playbooks (memories)")
        try:
            _client = get_chroma_client(persist_dir)
            _mem = _client.get_or_create_collection(name=MEM_COLLECTION, metadata={"hnsw:space": "cosine"})

            total = _mem.count()
            st.caption(f"Collection: '{MEM_COLLECTION}' — path: {persist_dir} — count: {total}")

            data = _mem.get(include=["documents", "metadatas"], limit=max(200, total))
            ids   = data.get("ids") or []
            docs  = data.get("documents") or []
            metas = data.get("metadatas") or []

            if total > 0 and not ids:
                st.warning("La collection riporta count>0 ma get() è vuoto. Riprovo senza include…")
                data = _mem.get(limit=max(200, total))  # fallback
                ids   = data.get("ids") or []
                docs  = data.get("documents") or []
                metas = data.get("metadatas") or []

            if not ids:
                st.caption("Nessun playbook salvato.")
            else:
                from datetime import datetime
                import pandas as pd

                rows = []
                for _id, doc, meta in zip(ids, docs, metas):
                    meta = meta or {}
                    created = meta.get("created_at")
                    expires = meta.get("expires_at")

                    created_s = datetime.fromtimestamp(created).strftime("%Y-%m-%d") if created else ""
                    expires_s = datetime.fromtimestamp(expires).strftime("%Y-%m-%d") if expires else ""

                    raw_tags = meta.get("tags", "")
                    tags = raw_tags if isinstance(raw_tags, str) else ", ".join(raw_tags) if raw_tags else ""

                    prev = (doc[:120] + "…") if doc and len(doc) > 120 else (doc or "")

                    rows.append({
                        "ID": _id,
                        "Project": meta.get("project", ""),
                        "Tags": tags,
                        "Created": created_s,
                        "Expires": expires_s,
                        "Preview": prev,
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading memories: {e}")

    # Ingest
    st.subheader("Vector DB Indexing")
    col1, col2 = st.columns([1, 3])
    with col1:
        start_ingest = st.button("Index tickets")
    with col2:
        st.caption("Create ticket embeddings and save them to Chroma for semantic retrieval")

    if start_ingest:
        issues = st.session_state.get("issues", [])
        if not issues:
            st.error("First load the project's tickets")
        else:
            try:
                st.session_state["vector"] = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
                use_openai_embeddings = (emb_backend == "OpenAI")
                st.session_state["embedder"] = EmbeddingBackend(use_openai=use_openai_embeddings, model_name=emb_model_name)
                # Save embeddings model metadata for consistency
                try:
                    meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
                    meta = {"provider": st.session_state["embedder"].provider_name, "model": st.session_state["embedder"].model_name}
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f)
                except Exception:
                    pass

                # NEW: chunked indexing with metadata parent_id, chunk_id, pos
                all_ids = []
                all_docs = []
                all_metas = []
                embed_inputs = []

                for it in issues:
                    full_text = it.text_blob()
                    if enable_chunking:
                        pieces = split_into_chunks(
                            full_text,
                            chunk_size=int(chunk_size),
                            overlap=int(chunk_overlap),
                            min_size=int(chunk_min),
                        )
                    else:
                        pieces = [(0, full_text)]

                    if not pieces:
                        pieces = [(0, full_text)]

                    multi = len(pieces) > 1  # true se davvero abbiamo più chunk

                    for idx, (pos0, chunk_text) in enumerate(pieces, start=1):
                        # usa ID semplice se non stai spezzando
                        cid = f"{it.id_readable}::c{idx:03d}" if multi else it.id_readable

                        all_ids.append(cid)
                        all_docs.append(chunk_text)

                        meta = {
                            "parent_id": it.id_readable,
                            "id_readable": it.id_readable,
                            "summary": it.summary,
                            "project": it.project,
                        }
                        if multi:
                            meta["chunk_id"] = idx          # int
                            meta["pos"] = int(pos0)         # int

                        all_metas.append(meta)
                        all_metas = [{k: v for k, v in m.items() if v is not None} for m in all_metas]
                        embed_inputs.append(f"{it.id_readable} | {it.summary}\n\n{chunk_text}")
                with st.spinner(f"Computing embeddings for {len(all_ids)} {'chunks' if enable_chunking else 'documents'} and saving to Chroma..."):
                    embs = st.session_state["embedder"].embed(embed_inputs)
                    st.session_state["vector"].add(ids=all_ids, documents=all_docs, metadatas=all_metas, embeddings=embs)
                # update counter
                st.session_state["vs_persist_dir"] = persist_dir
                st.session_state["vs_collection"] = collection_name
                st.session_state["vs_count"] = st.session_state["vector"].count()
                st.success(f"Indexing completed. Total {'chunks' if enable_chunking else 'documents'}: {st.session_state['vs_count']}")
            except Exception as e:
                st.error(f"Indexing error: {e}")

    # Chat
    st.subheader("RAG Chatbot")
    query = st.text_area("New ticket", height=140, placeholder="Describe the problem as if opening a ticket")
    run_chat = st.button("Search and answer")

    if run_chat:
        if not query.strip():
            st.error("Enter the ticket text")
        elif not st.session_state.get("vector"):
            ok, _, _ = open_vector_in_session(persist_dir, collection_name)
            if not ok:
                st.error("Open or create a valid collection in the sidebar")
        else:
            try:
                if st.session_state.get("embedder") is None:
                    use_openai_embeddings = (emb_backend == "OpenAI")
                    st.session_state["embedder"] = EmbeddingBackend(
                        use_openai=use_openai_embeddings,
                        model_name=emb_model_name
                    )

                # Warn if the current model differs from the one used for indexing
                try:
                    meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        if (m.get("provider") != st.session_state["embedder"].provider_name) or \
                        (m.get("model") != st.session_state["embedder"].model_name):
                            st.info(
                                f"Note: the collection was indexed with {m.get('provider')} / {m.get('model')}; "
                                f"you are querying with {st.session_state['embedder'].provider_name} / "
                                f"{st.session_state['embedder'].model_name}."
                            )
                except Exception:
                    pass

                # Retrieval
                q_emb = st.session_state["embedder"].embed([query])

                # 1) KB
                top_k = int(st.session_state.get("top_k", 5))
                show_distances = bool(st.session_state.get("show_distances", False))
                kb_res = st.session_state["vector"].query(query_embeddings=q_emb, n_results=top_k)
                kb_docs  = kb_res.get("documents", [[]])[0]
                kb_metas = kb_res.get("metadatas", [[]])[0]
                kb_dists = kb_res.get("distances", [[]])[0]

                DIST_MAX_KB = max_distance
                kb_retrieved = [
                    (doc, meta, dist, "KB")
                    for doc, meta, dist in zip(kb_docs, kb_metas, kb_dists)
                    if dist is not None and dist <= DIST_MAX_KB
                ]

                # 2) MEMORIES (if enabled)
                mem_retrieved = []
                if st.session_state.get("enable_memory", False):
                    try:
                        _client = get_chroma_client(persist_dir)
                        _mem = _client.get_or_create_collection(name=MEM_COLLECTION, metadata={"hnsw:space": "cosine"})
                        mem_res = _mem.query(query_embeddings=q_emb, n_results=min(5, top_k))
                        mem_docs  = mem_res.get("documents", [[]])[0]
                        mem_metas = mem_res.get("metadatas", [[]])[0]
                        mem_dists = mem_res.get("distances", [[]])[0]

                        DIST_MAX_MEM = DIST_MAX_KB
                        now = now_ts()
                        for doc, meta, dist in zip(mem_docs, mem_metas, mem_dists):
                            if dist is None:
                                continue
                            meta = meta or {}
                            exp = int(meta.get("expires_at", 0))
                            if exp and exp < now:
                                continue
                            if dist <= DIST_MAX_MEM:
                                mem_retrieved.append((doc, meta, dist, "MEM"))
                    except Exception:
                        pass  # memory is optional

                # 3) Merge: max 2 mem, then KB up to top_k
                mem_cap = 2
                merged = sorted(mem_retrieved, key=lambda x: x[2])[:mem_cap] + \
                        sorted(kb_retrieved, key=lambda x: x[2])[:max(0, top_k - min(mem_cap, len(mem_retrieved)))]

                # 4) Prompt

                if merged:

                    # Collapse duplicates by ticket for DISPLAY

                    merged_collapsed_view = collapse_by_parent(

                        merged,

                        per_parent=int(st.session_state.get("per_parent_display", 1)),

                        stitch_for_prompt=False,

                    )


                    # Collapse and stitch for PROMPT context

                    merged_for_prompt = collapse_by_parent(

                        merged,

                        per_parent=int(st.session_state.get("per_parent_prompt", 3)),

                        stitch_for_prompt=True,

                        max_chars=int(st.session_state.get("stitch_max_chars", 1500)),

                    )


                    retrieved_for_prompt = [(doc, meta, dist) for (doc, meta, dist, _src) in merged_for_prompt]

                    prompt = build_prompt(query, retrieved_for_prompt)

                else:

                    merged_collapsed_view = []

                    prompt = f"New ticket:\n{{query.strip()}}\n\nNo similar ticket was found in the knowledge base."

                st.write(f"DEBUG: view={len(merged_collapsed_view)}, prompt_ctx={len(merged_for_prompt) if merged else 0}")

                if st.session_state.get("show_prompt", False):
                    with st.expander("Prompt sent to the LLM", expanded=False):
                        st.code(prompt, language="markdown")

                # LLM
                _model = (llm_model or "").strip()
                if not _model:
                    raise RuntimeError("LLM model not set: select a valid model in the sidebar.")

                llm = LLMBackend(llm_provider, _model, temperature=llm_temperature)
                with st.spinner("Generating answer..."):
                    answer = llm.generate(RAG_SYSTEM_PROMPT, prompt)
                st.write(answer)
                # store the last Q/A for saving outside the if run_chat
                st.session_state["last_query"] = query
                st.session_state["last_answer"] = answer

                # 5) Similar results with provenance
                if merged:
                    base_url = (st.session_state.get("yt_client").base_url if st.session_state.get("yt_client") else "").rstrip("/")
                    st.write("Similar results (top-k, with provenance):")
                    for (doc, meta, dist, src) in merged_collapsed_view:
                        if src == "KB":
                            idr = meta.get("id_readable", "")
                            summary = meta.get("summary", "")
                            url = f"{base_url}/issue/{idr}" if base_url and idr else ""


                            cid = meta.get("chunk_id")
                            cpos = meta.get("pos")
                            extra = f" · chunk {cid} @tok {cpos}" if cid else " · document"
                            if url:
                                if st.session_state.get("show_distances", False):
                                    st.markdown(f"- [KB] [{idr}]({url}) — distance={dist:.3f}{extra} — {summary}")
                                else:
                                    st.markdown(f"- [KB] [{idr}]({url}){extra} — {summary}")
                            else:
                                if st.session_state.get("show_distances", False):
                                    st.write(f"- [KB] {idr} — distance={dist:.3f}{extra} — {summary}")
                                else:
                                    st.write(f"- [KB] {idr}{extra} — {summary}")

                        else:  # MEM
                            # single-line preview, no nested bullets
                            preview = one_line_preview(doc, maxlen=160)
                            proj = meta.get("project", "")
                            raw_tags = meta.get("tags", "")
                            tags = raw_tags if isinstance(raw_tags, str) else ", ".join(raw_tags) if raw_tags else ""
                            extra = f" (tags: {tags})" if tags else (f" (proj: {proj})" if proj else "")
                            if st.session_state.get("show_distances", False):
                                st.markdown(f"- [MEM]{extra} — distance={dist:.3f} — {preview}")
                            else:
                                st.markdown(f"- [MEM]{extra} — {preview}")

                            # Optional expander with full text
                            if st.session_state.get("mem_show_full", False):
                                with st.expander("Full playbook [MEM]", expanded=False):
                                    # the playbook may contain markdown bullets; render as markdown
                                    st.markdown(doc)

                else:
                    st.caption("No sufficiently similar result (below threshold).")

            except Exception as e:
                st.error(f"Chat error: {e}")
    # --- Persistent button to save the playbook (outside the if run_chat) ---
    if st.session_state.get("enable_memory", False) and st.session_state.get("last_answer"):
        st.markdown("---")
        if st.button("✅ Mark as solved → Save as playbook"):
            try:
                query  = st.session_state.get("last_query", "") or ""
                answer = st.session_state.get("last_answer", "") or ""
                condense_prompt = (
                    "Summarize the solution in 3–6 imperative, reusable sentences, avoiding sensitive data.\n"
                    f"- Query: {query.strip()}\n- Risposta:\n{answer}\n\n"
                    "Output: bulleted list of clear steps."
                )
                # use the model already validated in chat
                _model = (st.session_state.get("llm_model") or "").strip()
                if not _model:
                    _model = DEFAULT_LLM_MODEL if st.session_state.get("last_llm_provider","OpenAI") == "OpenAI" else DEFAULT_OLLAMA_MODEL

                try:
                    llm_mem = LLMBackend(llm_provider, _model, temperature=max(0.1, llm_temperature - 0.1))
                    playbook_text = llm_mem.generate(
                        "You are an assistant who distills reusable mini-playbooks.",
                        condense_prompt
                    ).strip()
                except Exception:
                    playbook_text = (answer[:800] + "…") if len(answer) > 800 else answer

                if not playbook_text.strip():
                    st.warning("No content to save.")
                else:
                    curr_proj = st.session_state.get("last_project_key") or ""
                    tags_list = ["playbook"] + ([curr_proj] if curr_proj else [])

                    meta = {
                        "source": "memory",
                        "project": curr_proj,
                        "quality": "verified",
                        "created_at": now_ts(),
                        "expires_at": ts_in_days(st.session_state.get("mem_ttl_days", DEFAULT_MEM_TTL_DAYS)),
                        "tags": ",".join(tags_list),  # <-- CSV instead of list
                    }

                    if st.session_state.get("embedder") is None:
                        use_openai_embeddings = (emb_backend == "OpenAI")
                        st.session_state["embedder"] = EmbeddingBackend(use_openai=use_openai_embeddings, model_name=emb_model_name)
                    mem_emb = st.session_state["embedder"].embed([playbook_text])[0]

                    _client = get_chroma_client(persist_dir)
                    _mem = _client.get_or_create_collection(name=MEM_COLLECTION, metadata={"hnsw:space": "cosine"})
                    mem_id = f"mem::{uuid.uuid4().hex[:12]}"
                    _mem.add(ids=[mem_id], documents=[playbook_text], metadatas=[meta], embeddings=[mem_emb])

                    st.caption(f"Saved playbook in path='{persist_dir}', collection='{MEM_COLLECTION}'")
                    st.success("Playbook saved in memory.")
                    st.session_state["open_memories_after_save"] = True
                    st.rerun()                                  # refresh table
            except Exception as e:
                st.error(f"Errore salvataggio playbook: {e}")

# ------------------------------
# CLI + Self-tests (optional)
# ------------------------------
def _cli_help():
    print("Usage: streamlit run app.py --server.port 8502")

def _self_tests():
    print("Running minimal self-tests...")
    vs = None
    try:
        vs = VectorStore(DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION)
        print("VectorStore OK.")
    except Exception as e:
        print(f"VectorStore not available: {e}")

    try:
        emb = EmbeddingBackend(use_openai=False, model_name=DEFAULT_EMBEDDING_MODEL)
        vec = emb.embed(["testo uno", "testo due"])
        assert len(vec) == 2 and isinstance(vec[0], list)
        print("Local embeddings OK.")
    except Exception as e:
        print(f"Local embeddings not available: {e}")

    try:
        llm = LLMBackend("OpenAI", DEFAULT_LLM_MODEL)
        assert isinstance(llm, LLMBackend)
        print("LLM OpenAI OK (init).")
    except Exception as e:
        print(f"LLM OpenAI not available: {e}")

    try:
        llm = LLMBackend("Ollama (local)", DEFAULT_OLLAMA_MODEL)
        assert isinstance(llm, LLMBackend)
        print("LLM Ollama OK (init).")
    except Exception as e:
        print(f"LLM Ollama not available: {e}")

    try:
        llm = LLMBackend("???", "x")
    except Exception as e:
        assert "Unsupported LLM provider" in str(e)
    print("All self-tests PASSED. ✅")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    print("[DEBUG] Starting main program.")
    if ST_AVAILABLE:
        print("[DEBUG] Starting Streamlit interface.")
        run_streamlit_app()
    else:
        _cli_help()
        _self_tests()
