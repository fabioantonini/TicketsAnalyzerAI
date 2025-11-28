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
    "enable_chunking", 
    "chunk_size", 
    "chunk_overlap", 
    "chunk_min", 
    "show_distances", 
    "top_k", 
    "collapse_duplicates", 
    "per_parent_display", 
    "per_parent_prompt", 
    "stitch_max_chars",
    "enable_memory",
    "mem_ttl_days",
    "mem_show_full",
    "show_memories",
    "prefs_enabled",
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

    # --- Fix persist_dir for cloud/local ---
    pd = (prefs.get("persist_dir") or "").strip()
    if not pd:
        prefs["persist_dir"] = DEFAULT_CHROMA_DIR
    else:
        if IS_CLOUD and not pd.startswith("/tmp/"):
            prefs["persist_dir"] = DEFAULT_CHROMA_DIR

    # --- Initialize session_state for all prefs keys (once only) ---
    for key, value in prefs.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Keep full prefs dict available in session
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

def _normalize_provider_label(p: str) -> str:
    p = (p or "").strip().lower()
    if "openai" in p:
        return "openai"
    if "sentence" in p:  # "Local (sentence-transformers)"
        return "sentence-transformers"
    return p or "unknown"

def get_index_embedder_info(persist_dir: str, collection_name: str):
    """
    Ritorna (provider, model, source) dove source ∈ {"file","chroma",None}.
    1) Tenta <persist_dir>/<collection_name>__meta.json con chiavi {"provider","model"}
    2) Fallback: metadata della collection Chroma (se presenti)
    """
    # 1) file <collection>__meta.json
    try:
        meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            prov = _normalize_provider_label(m.get("provider"))
            model = (m.get("model") or "").strip()
            if prov and model:
                return prov, model, "file"
    except Exception:
        pass

    # 2) metadata su Chroma (se salvati in fase di ingest)
    try:
        client = get_chroma_client(persist_dir)
        coll = client.get_collection(collection_name)
        md = (getattr(coll, "metadata", None) or {})  # alcune versioni mettono metadata qui
        prov = _normalize_provider_label(md.get("provider"))
        model = (md.get("model") or "").strip()
        if prov and model:
            return prov, model, "chroma"
    except Exception:
        pass

    return None, None, None

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
NEW_LABEL = "➕ Create new collection..."

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

def render_phase_yt_connection_page(prefs):
    import streamlit as st  # local import to keep signature simple

    st.title("Phase 1 – YouTrack connection")
    st.write("Configure the YouTrack endpoint used to fetch and index tickets.")

    if st.session_state.get("yt_client"):
        st.success("Connected to YouTrack")

    # Read current values (from session_state or prefs)
    current_url = st.session_state.get("yt_url", prefs.get("yt_url", ""))
    current_token = st.session_state.get("yt_token", "")

    col1, col2 = st.columns(2)
    with col1:
        yt_url = st.text_input(
            "YouTrack server URL",
            value=current_url,
            placeholder="https://<org>.myjetbrains.com/youtrack",
        )
    with col2:
        yt_token = st.text_input(
            "YouTrack token (Bearer)",
            type="password",
            value=current_token,
        )

    # Keep state in session_state (not in prefs, to avoid writing token to disk)
    st.session_state["yt_url"] = yt_url
    st.session_state["yt_token"] = yt_token

    connect = st.button("Connect to YouTrack")

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

    return yt_url, yt_token, connect

def render_phase_embeddings_vectordb_page(prefs):
    import streamlit as st

    # Normalize prefs to a dict
    if isinstance(prefs, dict):
        prefs_dict = prefs
    else:
        prefs_dict = st.session_state.get("prefs", {})

    st.title("Phase 2 – Embeddings & Vector DB")
    st.write(
        "Configure the Chroma vector store (path and collections) and the embeddings "
        "provider/model used for indexing and query."
    )

    # Small status summary on top (allineato con prefs + session_state)
    current_dir = (
        st.session_state.get("persist_dir")
        or prefs_dict.get("persist_dir", DEFAULT_CHROMA_DIR)
    )

    current_collection = (
        st.session_state.get("vs_collection")
        or prefs_dict.get("collection_selected")
        or prefs_dict.get("new_collection_name", "")
    )

    current_provider = (
        st.session_state.get("emb_provider_select")
        or prefs_dict.get("emb_backend", "OpenAI")
    )

    current_model = (
        st.session_state.get("emb_model")
        or prefs_dict.get("emb_model_name", "")
    )

    st.info(
        f"Current collection: **{current_collection or 'none'}** · "
        f"Provider: **{current_provider}** · Model: **{current_model or 'n/a'}**"
    )

    st.markdown("---")

    # -----------------------------
    # Vector DB configuration
    # -----------------------------
    st.header("Vector DB")

    persist_dir_default = prefs_dict.get("persist_dir", DEFAULT_CHROMA_DIR)
    persist_dir = st.text_input(
        "Chroma path",
        value=st.session_state.get("persist_dir", persist_dir_default),
        key="persist_dir",
        help="Directory where Chroma will store its collections.",
    )

    # Ensure directory exists
    try:
        os.makedirs(persist_dir, exist_ok=True)
        st.caption(f"Current Chroma path: {persist_dir}")
    except Exception as e:
        st.error(f"Failed to create directory '{persist_dir}': {e}")
        return

    # -----------------------------
    # Read existing collections
    # -----------------------------
    coll_options: list[str] = []
    try:
        if chromadb is not None:
            _client = get_chroma_client(persist_dir)
            coll_options = [c.name for c in _client.list_collections()]  # type: ignore
    except Exception as e:
        st.caption(f"Unable to read collections from '{persist_dir}': {e}")

    # -----------------------------
    # Keep last selection / new name
    # -----------------------------
    last_sel = st.session_state.get("collection_selected")
    if not last_sel:
        last_sel = (prefs_dict.get("collection_selected") or "").strip() or None

    last_new = st.session_state.get("new_collection_name_input")
    if not last_new:
        last_new = prefs_dict.get("new_collection_name", DEFAULT_COLLECTION)
        # initialize session_state once, then the widget will read from here
        st.session_state["new_collection_name_input"] = last_new

    st.caption(f"Selected pref: {prefs_dict.get('collection_selected', '—')}  ·  Path: {persist_dir}")

    # -----------------------------
    # Select / create collection
    # -----------------------------
    if coll_options:
        opts = coll_options + [NEW_LABEL]

        if last_sel in opts:
            default_index = opts.index(last_sel)
        elif DEFAULT_COLLECTION in opts:
            default_index = opts.index(DEFAULT_COLLECTION)
        else:
            default_index = 0

        sel = st.selectbox(
            "Collection",
            options=opts,
            index=default_index,
            key="collection_select",
        )

        if sel == NEW_LABEL:
            new_name = st.text_input("New collection name", key="new_collection_name_input")
            st.caption("ℹ️ This collection will be physically created after you run **Index tickets** at least once.")
            collection_name = (st.session_state.get("new_collection_name_input") or "").strip() or DEFAULT_COLLECTION            
            st.session_state["collection_selected"] = NEW_LABEL
            st.session_state["vs_collection"] = collection_name
        else:
            collection_name = sel
            st.session_state["collection_selected"] = sel
            st.session_state["vs_collection"] = collection_name
            st.session_state["after_delete_reset"] = True
    else:
        st.caption("No collection found. Create a new one:")
        new_name = st.text_input("New collection", key="new_collection_name_input")
        st.caption("ℹ️ This collection will be physically created after you run **Index tickets** at least once.")
        collection_name = (st.session_state.get("new_collection_name_input") or "").strip() or DEFAULT_COLLECTION
        st.session_state["collection_selected"] = NEW_LABEL
        st.session_state["vs_collection"] = collection_name

    # -----------------------------
    # Collection management: delete
    # -----------------------------
    st.markdown("---")
    st.subheader("Collection management")

    is_existing_collection = collection_name in coll_options

    del_confirm = st.checkbox(
        f"Confirm deletion of '{collection_name}'",
        value=False,
        disabled=not is_existing_collection,
        help="This operation permanently removes the collection from the datastore.",
    )

    if not is_existing_collection:
        st.caption(
            "ℹ️ This collection name is not yet present in Chroma. "
            "Run **Index tickets** at least once before it can be deleted."
        )

    if st.button(
        "Delete collection",
        type="secondary",
        disabled=not is_existing_collection,
        help="Permanently removes the selected collection from the vector datastore.",
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
                    st.session_state["vs_persist_dir"] = persist_dir

                # Clear any loaded results/issues (optional but recommended)
                st.session_state["issues"] = []

                # Remove selection and new name from session
                st.session_state["collection_selected"] = None
                st.session_state["after_delete_reset"] = True

                # Update sticky prefs, if present
                prefs_in_state = st.session_state.get("prefs", {})
                prefs_in_state["collection_selected"] = None
                prefs_in_state["new_collection_name"] = DEFAULT_COLLECTION
                st.session_state["prefs"] = prefs_in_state
                try:
                    save_prefs(prefs_in_state)
                except Exception:
                    pass

                st.success(f"Collection '{collection_name}' deleted successfully.")
                st.rerun()

            except Exception as e:
                st.error(f"Error during deletion: {e}")

    st.markdown("---")

    # -----------------------------
    # Embeddings configuration
    # -----------------------------
    st.header("Embeddings")

    emb_backends: list[str] = []
    if SentenceTransformer is not None and not IS_CLOUD:
        emb_backends.append("Local (sentence-transformers)")
    emb_backends.append("OpenAI")

    pref_backend = prefs_dict.get("emb_backend") or "OpenAI"
    if pref_backend == "Local (sentence-transformers)" and "Local (sentence-transformers)" not in emb_backends:
        pref_backend = "OpenAI"

    emb_backend = st.selectbox(
        "Embeddings provider",
        options=emb_backends,
        index=emb_backends.index(pref_backend),
        key="emb_provider_select",
    )

    # Reset the model ONLY if the user has just changed the provider
    prev_backend = st.session_state.get("_prev_emb_backend")

    if prev_backend is None:
        # First time: keep the existing model from prefs/session_state
        st.session_state["_prev_emb_backend"] = emb_backend

    elif prev_backend != emb_backend:
        # Real provider change → reset model accordingly
        st.session_state["_prev_emb_backend"] = emb_backend
        st.session_state["emb_model"] = (
            "all-MiniLM-L6-v2" if emb_backend == "Local (sentence-transformers)" else "text-embedding-3-small"
        )

    if "emb_model" not in st.session_state:
        st.session_state["emb_model"] = prefs_dict.get(
            "emb_model_name",
            "all-MiniLM-L6-v2" if emb_backend == "Local (sentence-transformers)" else "text-embedding-3-small",
        )

    emb_model_options = (
        ["all-MiniLM-L6-v2"]
        if emb_backend == "Local (sentence-transformers)"
        else ["text-embedding-3-small", "text-embedding-3-large"]
    )

    if st.session_state["emb_model"] not in emb_model_options:
        st.session_state["emb_model"] = emb_model_options[0]

    emb_model_name = st.selectbox(
        "Embeddings model",
        options=emb_model_options,
        index=emb_model_options.index(st.session_state["emb_model"]),
        key="emb_model",
    )

    # Ingest
    st.subheader("Vector DB Indexing")
    # --- Embeddings backend/model for ingestion, chat and playbooks ---
    emb_backend = st.session_state.get(
        "emb_provider_select",
        prefs.get("emb_backend", "OpenAI"),
    )
    emb_model_name = st.session_state.get(
        "emb_model",
        prefs.get("emb_model_name", "text-embedding-3-small"),
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        start_ingest = st.button("Index tickets")
    with col2:
        st.caption("Create ticket embeddings and save them to Chroma for semantic retrieval")

    # Read chunking configuration from session_state
    enable_chunking = bool(st.session_state.get("enable_chunking", True))
    chunk_size = int(st.session_state.get("chunk_size", 800))
    chunk_overlap = int(st.session_state.get("chunk_overlap", 80))
    chunk_min = int(st.session_state.get("chunk_min", 512))

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
                st.session_state["ui_phase_choice"] = "Embeddings & Vector DB"
                st.rerun()
            except Exception as e:
                st.error(f"Indexing error: {e}")

def render_phase_retrieval_page(prefs):
    import streamlit as st

    # Normalizza prefs in dict
    if isinstance(prefs, dict):
        prefs_dict = prefs
    else:
        prefs_dict = st.session_state.get("prefs", {})

    st.title("Phase 3 – Retrieval configuration")
    st.write(
        "Configure the distance threshold and chunking parameters used to retrieve "
        "similar tickets from the vector store."
    )

    st.markdown("### Distance threshold")

    max_default = float(prefs_dict.get("max_distance", 0.9))
    max_distance = st.slider(
        "Maximum distance threshold (cosine)",
        min_value=0.1,
        max_value=2.0,
        step=0.05,
        value=float(st.session_state.get("max_distance", max_default)),
        key="max_distance",
        help="Maximum cosine distance for a result to be considered similar."
    )
    # Nessuna scrittura manuale in session_state: ci pensa lo slider

    st.markdown("---")
    st.subheader("Chunking configuration")

    enable_default = bool(prefs_dict.get("enable_chunking", True))
    st.checkbox(
        "Enable chunking",
        key="enable_chunking",
        value=bool(st.session_state.get("enable_chunking", enable_default)),
        help="If enabled, long tickets are split into overlapping chunks before indexing."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.number_input(
            "Chunk size (tokens)",
            min_value=128,
            max_value=2048,
            step=64,
            key="chunk_size",
            value=int(st.session_state.get("chunk_size", prefs_dict.get("chunk_size", 800))),
            help="Typical values: 512–800."
        )

    with c2:
        st.number_input(
            "Overlap (tokens)",
            min_value=0,
            max_value=512,
            step=10,
            key="chunk_overlap",
            value=int(st.session_state.get("chunk_overlap", prefs_dict.get("chunk_overlap", 80))),
            help="How many tokens to overlap between consecutive chunks."
        )

    with c3:
        st.number_input(
            "Min size to chunk",
            min_value=128,
            max_value=2048,
            step=64,
            key="chunk_min",
            value=int(st.session_state.get("chunk_min", prefs_dict.get("chunk_min", 512))),
            help="Below this size, the ticket is indexed as a single document."
        )

    st.info(
        "These settings are used when you run indexing (Phase 2) and when the retriever "
        "filters similar tickets (Phase 6)."
    )

    # -----------------------------
    # Advanced settings reset (pre-widgets)
    # -----------------------------
    if st.session_state.pop("_adv_do_reset", False):
        _defaults = {
            # Advanced
            "adv_show_distances": False,
            "adv_top_k": 5,
            "adv_collapse_duplicates": True,
            "adv_per_parent_display": 1,
            "adv_per_parent_prompt": 3,
            "adv_stitch_max_chars": 1500,
            # Chunking
            "enable_chunking": True,
            "chunk_size": 800,
            "chunk_overlap": 80,
            "chunk_min": 512,
        }
        for _k, _v in _defaults.items():
            st.session_state[_k] = _v
        st.session_state["_adv_reset_toast"] = True

    # -----------------------------
    # Advanced settings (UI)
    # -----------------------------
    with st.expander("Advanced settings", expanded=False):
        show_distances = st.checkbox(
            "Show distances in results",
            key="adv_show_distances",
            help="Display the distance/score next to each retrieved result.",
        )

        top_k = st.number_input(
            "Top-K KB results",
            min_value=1,
            max_value=50,
            step=1,
            key="adv_top_k",
            help="Number of Knowledge Base results to pass downstream (before collapsing duplicates).",
        )

        collapse_duplicates = st.checkbox(
            "Collapse duplicate results by ticket",
            key="adv_collapse_duplicates",
            help="Show only one result per ticket in the UI (keeps recall for the prompt context).",
        )

        per_parent_display = st.number_input(
            "Max results per ticket (UI)",
            min_value=1,
            max_value=10,
            step=1,
            key="adv_per_parent_display",
            help="Maximum number of results displayed for the same ticket in the UI.",
        )

        per_parent_prompt = st.number_input(
            "Max chunks per ticket (prompt)",
            min_value=1,
            max_value=10,
            step=1,
            key="adv_per_parent_prompt",
            help="Maximum number of chunks concatenated per ticket in the prompt context.",
        )

        stitch_max_chars = st.number_input(
            "Stitched context limit (chars)",
            min_value=200,
            max_value=20000,
            step=100,
            key="adv_stitch_max_chars",
            help="Character limit when concatenating multiple chunks of the same ticket for the prompt context.",
        )

        if st.button("Reset to defaults", key="adv_reset_btn"):
            st.session_state["_adv_do_reset"] = True
            st.rerun()

    if st.session_state.pop("_adv_reset_toast", False):
        st.toast("Advanced settings reset to defaults", icon="↩️")

    # Make them available during the current run (for downstream functions)
    st.session_state["show_distances"] = st.session_state.get("adv_show_distances", False)
    st.session_state["top_k"] = int(st.session_state.get("adv_top_k", 5))
    st.session_state["collapse_duplicates"] = st.session_state.get("adv_collapse_duplicates", True)
    st.session_state["per_parent_display"] = int(st.session_state.get("adv_per_parent_display", 1))
    st.session_state["per_parent_prompt"] = int(st.session_state.get("adv_per_parent_prompt", 3))
    st.session_state["stitch_max_chars"] = int(st.session_state.get("adv_stitch_max_chars", 1500))

def render_phase_llm_page(prefs):
    import streamlit as st

    # Normalizza prefs in dict per avere default decenti
    if isinstance(prefs, dict):
        prefs_dict = prefs
    else:
        prefs_dict = st.session_state.get("prefs", {})

    st.title("Phase 4 – LLM & API keys")
    st.write(
        "Configure the LLM provider, model, temperature and API keys used to answer "
        "questions based on the retrieved tickets."
    )

    # Piccolo riassunto in alto
    current_provider = st.session_state.get(
        "llm_provider_select",
        prefs_dict.get("llm_provider", "OpenAI"),
    )
    current_model = st.session_state.get(
        "llm_model",
        prefs_dict.get("llm_model", "gpt-4o"),
    )
    current_temp = float(
        st.session_state.get(
            "llm_temperature",
            prefs_dict.get("llm_temperature", 0.2),
        )
    )

    st.info(
        f"Current LLM: **{current_provider} · {current_model}** · "
        f"Temperature: **{current_temp:.2f}**"
    )

    st.markdown("---")

    st.header("LLM")

    # Ollama detection
    ollama_ok, ollama_host = (False, None) if IS_CLOUD else is_ollama_available()
    llm_provider_options = ["OpenAI"] + (["Ollama (local)"] if ollama_ok else [])

    # Default index consistent with prefs but safe if Ollama is NOT available
    pref_provider = prefs_dict.get("llm_provider", "OpenAI")
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
    pref_llm_model = (prefs_dict.get("llm_model") or "").strip()

    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = pref_llm_model or llm_model_default
    else:
        if not (st.session_state["llm_model"] or "").strip():
            st.session_state["llm_model"] = llm_model_default

    # Field controlled via session_state
    llm_model = st.text_input("LLM model", key="llm_model")

    # Temperature slider (as in your code)
    if "llm_temperature" not in st.session_state:
        st.session_state["llm_temperature"] = float(prefs_dict.get("llm_temperature", 0.2))
    llm_temperature = st.slider("Temperature", 0.0, 1.5, st.session_state["llm_temperature"], 0.05)
    st.session_state["llm_temperature"] = llm_temperature

    # --- [SIDEBAR > API Keys] ---
    st.header("API Keys")
    # Read providers from session_state (fallback to defaults / prefs)
    emb_backend = st.session_state.get(
        "emb_provider_select",
        prefs_dict.get("emb_backend", "OpenAI"),
    )

    llm_provider = st.session_state.get(
        "llm_provider_select",
        prefs_dict.get("llm_provider", "OpenAI"),
    )

    openai_needed = (emb_backend == "OpenAI") or (llm_provider == "OpenAI")

    openai_key_input = st.text_input("OpenAI API Key",
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

def render_phase_chat_page(prefs):
    import streamlit as st

    # Normalize prefs to a dict
    if isinstance(prefs, dict):
        prefs_dict = prefs
    else:
        prefs_dict = st.session_state.get("prefs", {})

    # Persist dir and collection from session_state (with prefs fallback)
    persist_dir = st.session_state.get(
        "persist_dir",
        prefs_dict.get("persist_dir", DEFAULT_CHROMA_DIR),
    )

    collection_name = st.session_state.get(
        "vs_collection",
        st.session_state.get("new_collection_name_input", DEFAULT_COLLECTION),
    )

    max_distance = float(
        st.session_state.get("max_distance", prefs_dict.get("max_distance", 0.9))
    )

    # Embeddings backend/model for retrieval + memory
    emb_backend = st.session_state.get(
        "emb_provider_select",
        prefs_dict.get("emb_backend", "OpenAI"),
    )
    emb_model_name = st.session_state.get(
        "emb_model",
        prefs_dict.get("emb_model_name", "text-embedding-3-small"),
    )

    # LLM provider/model/temperature for chat + playbooks
    llm_provider = st.session_state.get(
        "llm_provider_select",
        prefs_dict.get("llm_provider", "OpenAI"),
    )
    llm_temperature = float(
        st.session_state.get("llm_temperature", prefs_dict.get("llm_temperature", 0.2))
    )
    llm_model = st.session_state.get(
        "llm_model",
        prefs_dict.get(
            "llm_model",
            DEFAULT_LLM_MODEL if llm_provider == "OpenAI" else DEFAULT_OLLAMA_MODEL,
        ),
    )

    st.title("Phase 5 – Chat & Results")
    st.write(
        "Ask questions about your YouTrack tickets. The app retrieves similar tickets "
        "from the vector store, sends them to the LLM, and shows the answer together "
        "with the retrieved context."
    )

    st.markdown("---")

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

                max_distance = float(st.session_state.get("max_distance", prefs.get("max_distance", 0.9)))
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

                    prompt = f"New ticket:\n{query.strip()}\n\nNo similar ticket was found in the knowledge base."

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

def render_phase_solutions_memory_page(prefs):
    import streamlit as st

    # 1) prefs_dict: dai priorità a quelli in session_state
    if isinstance(prefs, dict):
        prefs_dict = st.session_state.get("prefs", prefs)
    else:
        prefs_dict = st.session_state.get("prefs", {})

    # 2) Persist dir per la collection dei playbook
    persist_dir = st.session_state.get(
        "persist_dir",
        prefs_dict.get("persist_dir", DEFAULT_CHROMA_DIR),
    )

    # 3) Bootstrap UNA SOLA VOLTA i valori in session_state dai prefs
    if "enable_memory" not in st.session_state:
        st.session_state["enable_memory"] = bool(prefs_dict.get("enable_memory", False))

    if "mem_ttl_days" not in st.session_state:
        st.session_state["mem_ttl_days"] = int(prefs_dict.get("mem_ttl_days", DEFAULT_MEM_TTL_DAYS))

    if "mem_show_full" not in st.session_state:
        st.session_state["mem_show_full"] = bool(prefs_dict.get("mem_show_full", False))

    if "show_memories" not in st.session_state:
        st.session_state["show_memories"] = bool(prefs_dict.get("show_memories", False))

    st.title("Phase 6 – Solutions memory")
    st.write(
        "Review and manage saved playbooks (memories) derived from solved tickets."
    )

    st.markdown("---")
    st.header("Solutions memory")

    # 4) Widget → value da session_state, poi riscrivo in session_state
    mem_show_full = st.checkbox(
        "Show full text of playbooks (MEM)",
        value=st.session_state["mem_show_full"],
        help="If enabled, an expander with the full playbook appears under each MEM result.",
    )
    st.session_state["mem_show_full"] = mem_show_full

    enable_memory = st.checkbox(
        "Enable 'playbook' memory (separate collection)",
        value=st.session_state["enable_memory"],
        help="When you mark an answer as 'Solved', I save a reusable mini-playbook.",
    )
    st.session_state["enable_memory"] = enable_memory

    mem_ttl_days = st.number_input(
        "TTL (days) for playbooks",
        min_value=7,
        max_value=365,
        step=1,
        value=int(st.session_state["mem_ttl_days"]),
    )
    st.session_state["mem_ttl_days"] = int(mem_ttl_days)

    c_mem1, c_mem2 = st.columns(2)
    with c_mem1:
        if st.session_state.pop("open_memories_after_save", False):
            st.session_state["show_memories"] = True

        show_memories = st.checkbox(
            "Show saved playbooks",
            value=st.session_state["show_memories"],
            help="Display the list of the 'memories' collection on the main page.",
        )
        st.session_state["show_memories"] = show_memories

    with c_mem2:
        mem_del_confirm = st.checkbox("Confirm delete memories", value=False)
        if st.button("Delete all memories", disabled=not mem_del_confirm):
            try:
                _client = get_chroma_client(persist_dir)
                _client.delete_collection(name=MEM_COLLECTION)
                _client.get_or_create_collection(
                    name=MEM_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
                st.success("Memories deleted.")
            except Exception as e:
                st.error(f"Error deleting memories: {e}")

    st.markdown("---")

    # 5) Tabella dei playbook solo se attivata
    if st.session_state.get("show_memories"):
        st.subheader("Saved playbooks (memories)")
        try:
            _client = get_chroma_client(persist_dir)
            _mem = _client.get_or_create_collection(
                name=MEM_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

            total = _mem.count()
            st.caption(
                f"Collection: '{MEM_COLLECTION}' — path: {persist_dir} — count: {total}"
            )

            data = _mem.get(include=["documents", "metadatas"], limit=max(200, total))
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []

            if total > 0 and not ids:
                st.warning("La collection riporta count>0 ma get() è vuoto. Riprovo senza include…")
                data = _mem.get(limit=max(200, total))  # fallback
                ids = data.get("ids") or []
                docs = data.get("documents") or []
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
                    if isinstance(raw_tags, str):
                        tags = raw_tags
                    elif raw_tags:
                        tags = ", ".join(raw_tags)
                    else:
                        tags = ""

                    prev = (doc[:120] + "…") if doc and len(doc) > 120 else (doc or "")

                    rows.append(
                        {
                            "ID": _id,
                            "Project": meta.get("project", ""),
                            "Tags": tags,
                            "Created": created_s,
                            "Expires": expires_s,
                            "Preview": prev,
                        }
                    )

                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading memories: {e}")

def render_phase_preferences_debug_page(prefs):
    st.title("Phase 7 – Preferences & debug")
    st.write(
        "Preferences handling and Debug settings"
    )
    st.subheader("Preferences")

    # Normalize prefs to a dict
    if isinstance(prefs, dict):
        prefs_dict = prefs
    else:
        prefs_dict = st.session_state.get("prefs", {})

    if "prefs_enabled" not in st.session_state:
        st.session_state["prefs_enabled"] = bool(prefs_dict.get("prefs_enabled", True))

    prefs_enabled = st.checkbox(
        "Enable preferences memory (local)",
        value=st.session_state["prefs_enabled"],
    )
    st.session_state["prefs_enabled"] = int(prefs_enabled)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save preferences"):
            if prefs_enabled:
                try:
                    # --- Provider coercion: if Ollama not available, force OpenAI ---
                    _provider_for_save = st.session_state.get("last_llm_provider") or \
                        st.session_state.get("llm_provider_select") or prefs_dict.get("llm_provider", "OpenAI")

                    # Local Ollama availability check (do not rely on UI-phase variable)
                    if IS_CLOUD:
                        _ollama_ok = False
                    else:
                        _ollama_ok, _ = is_ollama_available()
                    if not _ollama_ok and _provider_for_save == "Ollama (local)":
                        _provider_for_save = "OpenAI"

                    # --- LLM model: never empty and consistent with provider ---
                    _model_for_save = (st.session_state.get("llm_model") or "").strip()
                    if not _model_for_save:
                        _model_for_save = DEFAULT_LLM_MODEL if _provider_for_save == "OpenAI" else DEFAULT_OLLAMA_MODEL

                    # --- Read current UI values or session defaults ---
                    yt_url = st.session_state.get("yt_url", prefs_dict.get("yt_url", ""))
                    persist_dir = st.session_state.get("persist_dir", prefs_dict.get("persist_dir", ""))

                    emb_backend = st.session_state.get("emb_provider_select", prefs_dict.get("emb_backend", "OpenAI"))
                    emb_model_name = st.session_state.get("emb_model", prefs_dict.get("emb_model_name", "text-embedding-3-small"))

                    llm_temperature = st.session_state.get("llm_temperature", prefs_dict.get("llm_temperature", 0.2))
                    max_distance = st.session_state.get("max_distance", prefs_dict.get("max_distance", 0.9))
                    show_prompt = st.session_state.get("show_prompt", prefs_dict.get("show_prompt", True))
                    collection_selected = st.session_state.get("collection_selected", prefs_dict.get("collection_selected"))
                    new_collection_name = st.session_state.get("new_collection_name_input", prefs_dict.get("new_collection_name", "tickets"))

                    # --- Chunking ---
                    enable_chunking = bool(st.session_state.get("enable_chunking", prefs_dict.get("enable_chunking", True)))
                    chunk_size = int(st.session_state.get("chunk_size", prefs_dict.get("chunk_size", 800)))
                    chunk_overlap = int(st.session_state.get("chunk_overlap", prefs_dict.get("chunk_overlap", 80)))
                    chunk_min = int(st.session_state.get("chunk_min", prefs_dict.get("chunk_min", 512)))

                    # --- Advanced settings ---
                    show_distances = bool(st.session_state.get("adv_show_distances", prefs_dict.get("show_distances", False)))
                    top_k = int(st.session_state.get("adv_top_k", prefs_dict.get("top_k", 5)))
                    collapse_duplicates = bool(st.session_state.get("adv_collapse_duplicates", prefs_dict.get("collapse_duplicates", True)))
                    per_parent_display = int(st.session_state.get("adv_per_parent_display", prefs_dict.get("per_parent_display", 1)))
                    per_parent_prompt = int(st.session_state.get("adv_per_parent_prompt", prefs_dict.get("per_parent_prompt", 3)))
                    stitch_max_chars = int(st.session_state.get("adv_stitch_max_chars", prefs_dict.get("stitch_max_chars", 1500)))

                    # --- Solutions memory settings ---
                    enable_memory = bool(st.session_state.get("enable_memory", prefs_dict.get("enable_memory", False)))
                    mem_ttl_days = int(st.session_state.get("mem_ttl_days", prefs_dict.get("mem_ttl_days", DEFAULT_MEM_TTL_DAYS)))
                    mem_show_full = bool(st.session_state.get("mem_show_full", prefs_dict.get("mem_show_full", False)))
                    show_memories = bool(st.session_state.get("show_memories", prefs_dict.get("show_memories", False)))

                    # --- Save all prefs ---
                    new_prefs = {
                        "yt_url": yt_url,
                        "persist_dir": persist_dir,
                        "emb_backend": emb_backend,
                        "emb_model_name": emb_model_name,
                        "llm_provider": _provider_for_save,
                        "llm_model": _model_for_save,
                        "llm_temperature": llm_temperature,
                        "max_distance": max_distance,
                        "show_prompt": show_prompt,
                        "collection_selected": collection_selected,
                        "new_collection_name": new_collection_name,
                        "enable_chunking": enable_chunking,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunk_min": chunk_min,
                        "show_distances": show_distances,
                        "top_k": top_k,
                        "collapse_duplicates": collapse_duplicates,
                        "per_parent_display": per_parent_display,
                        "per_parent_prompt": per_parent_prompt,
                        "stitch_max_chars": stitch_max_chars,
                        "enable_memory": enable_memory,
                        "mem_ttl_days": mem_ttl_days,
                        "mem_show_full": mem_show_full,
                        "show_memories": show_memories,
                        "prefs_enabled": prefs_enabled,
                    }

                    save_prefs(new_prefs)
                    st.session_state["prefs"] = load_prefs()
                    st.success("Preferences saved.")
                    st.toast("Saved to .app_prefs.json", icon="✅")

                except Exception as e:
                    import traceback, textwrap
                    st.error(f"Errore durante il salvataggio: {e}")
                    st.code(textwrap.dedent(traceback.format_exc()))
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

    st.header("Debug")
    if "show_prompt" not in st.session_state:
        st.session_state["show_prompt"] = bool(prefs_dict.get("show_prompt", False))
    show_prompt = st.checkbox("Show LLM prompt", value=st.session_state["show_prompt"])
    st.session_state["show_prompt"] = show_prompt

# ------------------------------
# Main Streamlit UI
# ------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="YouTrack RAG Support", layout="wide")
    init_prefs_in_session()
    # === Robust prefs loading & one-time bootstrap ===
    DEFAULT_PREFS = {
        "yt_url": "",
        "persist_dir": "",
        "emb_backend": "OpenAI",
        "emb_model_name": "text-embedding-3-small",
        "llm_provider": "OpenAI",
        "llm_model": "gpt-4o",
        "llm_temperature": 0.2,
        "max_distance": 0.9,
        "show_prompt": True,
        "collection_selected": None,
        "new_collection_name": "tickets",

        # chunking
        "enable_chunking": True,
        "chunk_size": 800,
        "chunk_overlap": 80,
        "chunk_min": 512,

        # advanced settings
        "show_distances": False,
        "top_k": 5,
        "collapse_duplicates": True,
        "per_parent_display": 1,
        "per_parent_prompt": 3,
        "stitch_max_chars": 1500,

        # solutions memory (NUOVI)
        "enable_memory": False,
        "mem_ttl_days": DEFAULT_MEM_TTL_DAYS,
        "mem_show_full": False,
        "show_memories": False,
    }

    def _normalize(p: dict) -> dict:
        p = p or {}
        norm = DEFAULT_PREFS.copy()
        norm.update(p)
        return norm

    # Bootstrap prefs -> session una sola volta
    if not st.session_state.get("_prefs_bootstrapped", False):
        prefs = _normalize(st.session_state.get("prefs", {}))

        # Inizializza SOLO se mancanti (non sovrascrivere scelte dell'utente)
        def _init(key, val):
            if key not in st.session_state:
                st.session_state[key] = val

        # Chunking
        _init("enable_chunking", prefs.get("enable_chunking"))
        _init("chunk_size",       int(prefs.get("chunk_size", 800)))
        _init("chunk_overlap",    int(prefs.get("chunk_overlap", 80)))
        _init("chunk_min",        int(prefs.get("chunk_min", 512)))

        # Advanced
        _init("adv_show_distances",      prefs.get("show_distances"))
        _init("adv_top_k",               int(prefs.get("top_k", 5)))
        _init("adv_collapse_duplicates", prefs.get("collapse_duplicates"))
        _init("adv_per_parent_display",  int(prefs.get("per_parent_display", 1)))
        _init("adv_per_parent_prompt",   int(prefs.get("per_parent_prompt", 3)))
        _init("adv_stitch_max_chars",    int(prefs.get("stitch_max_chars", 1500)))

        # Nome nuova collection (solo se non c'è già)
        _init("new_collection_name_input", (prefs.get("new_collection_name") or DEFAULT_COLLECTION))

        # Solutions memory
        _init("enable_memory",     bool(prefs.get("enable_memory", False)))
        _init("mem_ttl_days",      int(prefs.get("mem_ttl_days", DEFAULT_MEM_TTL_DAYS)))
        _init("mem_show_full",     bool(prefs.get("mem_show_full", False)))
        _init("show_memories",     bool(prefs.get("show_memories", False)))

        st.session_state["_prefs_bootstrapped"] = True

    # Ensure prefs is available beyond the bootstrap block
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
    phase = st.session_state.get("ui_phase_choice", "YouTrack connection")

    # Default values, in case we are not on this phase
    yt_url = st.session_state.get("yt_url", prefs.get("yt_url", ""))
    yt_token = st.session_state.get("yt_token", "")
    connect = False

    if phase == "YouTrack connection":
        yt_url, yt_token, connect = render_phase_yt_connection_page(prefs)

    # Phase 2 – Embeddings & Vector DB (page)
    if phase == "Embeddings & Vector DB":
        render_phase_embeddings_vectordb_page(prefs)

    # Phase 3 – Retrieval configuration (page)
    if phase == "Retrieval configuration":
        render_phase_retrieval_page(prefs)

    # Phase 4 – LLM & API keys (page)
    if phase == "LLM & API keys":
        render_phase_llm_page(prefs)

    # Phase 5 – Chat & Results (page)
    if phase == "Chat & Results":
        render_phase_chat_page(prefs)

    # Phase 6 - Solutions memory (page)
    if phase == "Solutions memory":
        render_phase_solutions_memory_page(prefs)

    # Phase 7 - Preferences and debug (page)
    if phase == "Preferences & debug":
        render_phase_preferences_debug_page(prefs)

    with st.sidebar:
        # --- Wizard-style navigation (visual only, all sections still visible) ---
        PHASES = [
            "New RAG setup",
            "YouTrack connection",
            "Embeddings & Vector DB",
            "Retrieval configuration",
            "LLM & API keys",
            "Chat & Results",
            "Solutions memory",
            "Preferences & debug",
        ]

        if "ui_phase_choice" not in st.session_state:
            st.session_state["ui_phase_choice"] = PHASES[0]

        # Default index: keep last selection if present, otherwise 0
        default_idx = int(st.session_state.get("ui_phase_index", 0))
        if default_idx < 0 or default_idx >= len(PHASES):
            default_idx = 0

        current_phase = st.radio(
            "Current phase",
            options=PHASES,
            key="ui_phase_choice",
            help="This is only a visual wizard for now.",
        )

        if isinstance(current_phase, str):
            st.session_state["ui_phase_index"] = PHASES.index(current_phase)

        # Simple progress bar for the wizard
        step_idx = st.session_state.get("ui_phase_index", 0)
        st.progress((step_idx + 1) / len(PHASES), text=f"Step {step_idx + 1} / {len(PHASES)}")

        st.markdown("---")

        quit_btn = False
        st.markdown("### YouTrack status")
        if st.session_state.get("yt_client"):
            st.success("Connected to YouTrack")
        else:
            st.warning("Not connected to YouTrack")

        st.caption(st.session_state.get("yt_url", prefs.get("yt_url", "")) or "No URL configured")

        st.markdown("---")
        st.markdown("### Vector DB / Embeddings status")

        persist_dir = st.session_state.get("persist_dir", prefs.get("persist_dir", ""))
        vs_collection = st.session_state.get("vs_collection", "")
        emb_provider = st.session_state.get("emb_provider_select", "OpenAI")
        emb_model = st.session_state.get("emb_model", "")

        if vs_collection:
            st.success(f"Collection: {vs_collection}")
        else:
            st.warning("No collection selected")

        st.caption(persist_dir or "No Chroma path configured")
        st.caption(f"Embeddings: {emb_provider} · {emb_model or 'n/a'}")

        st.markdown("---")
        st.markdown("### LLM status")

        llm_provider = st.session_state.get("llm_provider_select", "OpenAI")
        llm_model = st.session_state.get("llm_model", "gpt-4o")
        llm_temperature = float(st.session_state.get("llm_temperature", 0.2))

        st.caption(f"Provider: {llm_provider}")
        st.caption(f"Model: {llm_model}")
        st.caption(f"Temperature: {llm_temperature:.2f}")

        # --- Retrieval summary (read-only) ---
        st.markdown("---")
        st.header("Retrieval summary")

        def _yes(v: bool) -> str:
            return "✅" if bool(v) else "✖️"

        st.markdown("---")
        st.markdown("### Chat status")

        last_query = st.session_state.get("last_query", "").strip()
        if last_query:
            st.caption(f"Last query: {last_query[:50]}{'…' if len(last_query) > 50 else ''}")
        else:
            st.caption("No query asked yet.")

        # Gather values from session/prefs
        _top_k   = int(st.session_state.get("top_k", 5))
        _maxdist = float(st.session_state.get("max_distance", 0.9))
        _collapse = bool(st.session_state.get("collapse_duplicates", True))
        _pp_ui   = int(st.session_state.get("per_parent_display", 1))
        _pp_pr   = int(st.session_state.get("per_parent_prompt", 3))
        _stitch  = int(st.session_state.get("stitch_max_chars", 1500))

        _chunk_on = bool(st.session_state.get("enable_chunking", True))
        _csize    = int(st.session_state.get("chunk_size", 800))
        _coverlap = int(st.session_state.get("chunk_overlap", 80))
        _cmin     = int(st.session_state.get("chunk_min", 512))

        _emb_backend = st.session_state.get("emb_provider_select") or st.session_state.get("prefs", {}).get("emb_backend", "OpenAI")
        _emb_model   = st.session_state.get("emb_model") or st.session_state.get("prefs", {}).get("emb_model_name", "text-embedding-3-small")

        _collection  = (
            st.session_state.get("collection_selected")
            or st.session_state.get("new_collection_name_input")
            or st.session_state.get("prefs", {}).get("new_collection_name")
            or "—"
        )

        # Row 1: core retrieval knobs
        c1, c2, c3 = st.columns(3)
        with c1: st.caption("Top-K"); st.write(_top_k)
        with c2: st.caption("Max distance"); st.write(f"{_maxdist:.2f}")
        with c3: st.caption("Collapse duplicates"); st.write(_yes(_collapse))

        # Row 2: per-ticket aggregation
        c4, c5, c6 = st.columns(3)
        with c4: st.caption("Per ticket (UI)"); st.write(_pp_ui)
        with c5: st.caption("Per ticket (prompt)"); st.write(_pp_pr)
        with c6: st.caption("Stitch limit (chars)"); st.write(_stitch)

        # Row 3: chunking
        c7, c8, c9, c10 = st.columns(4)
        with c7: st.caption("Chunking enabled"); st.write(_yes(_chunk_on))
        with c8: st.caption("Chunk size"); st.write(_csize)
        with c9: st.caption("Overlap"); st.write(_coverlap)
        with c10: st.caption("Min to chunk"); st.write(_cmin)

        # Row 4: embeddings + collection
        c11, c12 = st.columns(2)
        with c11: st.caption("Embeddings"); st.write(f"{_emb_backend} · {_emb_model}")
        with c12: st.caption("Collection"); st.write(_collection)

        # --- Embedding status (indexed vs query) ---
        _collection = (
            st.session_state.get("collection_selected")
            or st.session_state.get("new_collection_name_input")
            or st.session_state.get("prefs", {}).get("new_collection_name")
        )

        _persist_dir = st.session_state.get("persist_dir") or st.session_state.get("prefs", {}).get("persist_dir", "")

        # Embedder scelto per la QUERY (UI corrente)
        qry_provider = st.session_state.get("emb_provider_select") or st.session_state.get("prefs", {}).get("emb_backend", "OpenAI")
        qry_model    = st.session_state.get("emb_model") or st.session_state.get("prefs", {}).get("emb_model_name", "text-embedding-3-small")

        # Embedder con cui la COLLECTION è stata indicizzata (se noto)
        idx_provider, idx_model, idx_src = (None, None, None)
        if _collection and _persist_dir:
            idx_provider, idx_model, idx_src = get_index_embedder_info(_persist_dir, _collection)

        st.markdown("---")
        st.header("Embedding status")

        def _fmt(v): return v if v else "—"

        colA, colB = st.columns(2)
        with colA:
            st.caption("Indexed with")
            st.write(f"{_fmt(idx_provider)} · {_fmt(idx_model)}" + (f" ({idx_src})" if idx_src else ""))

        with colB:
            st.caption("Query using")
            st.write(f"{_fmt(qry_provider)} · {_fmt(qry_model)}")

        # Avviso mismatch (provider o modello diverso)
        if idx_provider and idx_model:
            mismatch = (_normalize_provider_label(qry_provider) != _normalize_provider_label(idx_provider)) or (qry_model.strip() != idx_model.strip())
            if mismatch:
                st.warning("Embedding mismatch between indexed collection and current query settings.", icon="⚠️")
        else:
            st.caption("No index metadata available. Consider reindexing to record provider/model.")

        st.markdown("---")
        st.caption(f"IS_CLOUD={IS_CLOUD} · ChromaDB dir={st.session_state['prefs'].get('persist_dir', DEFAULT_CHROMA_DIR)}")

        if not IS_CLOUD:
            quit_btn = st.button("Quit", use_container_width=True)
            if quit_btn:
                st.write("Closing application...")
                os._exit(0)

        # Automatic opening of the selected collection (without re-indexing)
        # Avoid opening if the user has chosen "Create new…" but has not entered a name different from the default
        vector_ready = False
        # Read persist_dir and collection name from session_state
        persist_dir = st.session_state.get(
            "persist_dir",
            prefs.get("persist_dir", DEFAULT_CHROMA_DIR) if isinstance(prefs, dict) else DEFAULT_CHROMA_DIR,
        )

        collection_name = (
            st.session_state.get("vs_collection")
            or st.session_state.get("new_collection_name_input", DEFAULT_COLLECTION)
        )

        if persist_dir and collection_name:
            changed = (
                st.session_state.get("vs_persist_dir") != persist_dir
                or st.session_state.get("vs_collection") != collection_name
                or st.session_state.get("vector") is None
            )

            # Do not open if we are on NEW_LABEL and the name is empty (or only default) and does not yet exist
            if st.session_state.get("collection_selected") == NEW_LABEL and (collection_name == "" or collection_name == DEFAULT_COLLECTION):
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
                with st.spinner("Loading projects..."):
                    st.session_state["projects"] = st.session_state["yt_client"].list_projects()
                st.rerun()
            except Exception as e:
                st.error(f"Connection failed: {e}")

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