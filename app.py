# app.py
"""
Streamlit RAG Support App (completa, con debug + self-tests + UI + Quit)

Funzionalità
1) Connessione a YouTrack (URL + token) e lista progetti/issues
2) Ingestione dei ticket in un Vector DB locale (Chroma) con embeddings locali o OpenAI
3) Selezione LLM (OpenAI o locale via Ollama)
4) Chat RAG: ricerca ticket simili + risposta LLM con citazioni ID
5) Modalità CLI con self-tests quando Streamlit non è disponibile
6) Pulsante Quit nella sidebar per chiudere l’app dal browser
7) Se trova data/chroma/chroma.sqlite3, la collezione viene aperta automaticamente e mostrato il numero di documenti caricati (o “N/A” se non disponibile)

Avvio consigliato
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

# === Sticky prefs (Livello A) ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PREFS_PATH = os.path.join(APP_DIR, ".app_prefs.json")  # file locale, NON contiene segreti

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
        import json, os
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: v for k, v in data.items() if k in NON_SENSITIVE_PREF_KEYS}
    except Exception:
        pass
    return {}

def save_prefs(prefs: dict):
    try:
        import json
        clean = {k: v for k, v in prefs.items() if k in NON_SENSITIVE_PREF_KEYS}
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] save_prefs: {e}")

def init_prefs_in_session():
    # carica da file una sola volta
    if "prefs_loaded" not in st.session_state:
        st.session_state["prefs_loaded"] = True
        st.session_state["prefs"] = load_prefs()

# ------------------------------
# Try imports con fallback/shim
# ------------------------------
try:
    import streamlit as st
    ST_AVAILABLE = True
    print("[DEBUG] Streamlit importato.")
except Exception:
    ST_AVAILABLE = False
    print("[DEBUG] Streamlit non disponibile, uso shim.")

    class _STShim:
        def __getattr__(self, _name):
            def _noop(*_args, **_kwargs):
                print(f"[DEBUG] Chiamata st.{_name} ignorata (shim).")
                return None
            return _noop

    st = _STShim()  # type: ignore

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    print("[DEBUG] chromadb importato.")
except Exception:
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore
    print("[DEBUG] chromadb non disponibile.")

try:
    from sentence_transformers import SentenceTransformer
    print("[DEBUG] sentence-transformers importato.")
except Exception:
    SentenceTransformer = None  # type: ignore
    print("[DEBUG] sentence-transformers non disponibile.")

try:
    from openai import OpenAI
    print("[DEBUG] openai SDK importato.")
except Exception:
    OpenAI = None  # type: ignore
    print("[DEBUG] openai SDK non disponibile.")


# ------------------------------
# Costanti
# ------------------------------
DEFAULT_CHROMA_DIR = "data/chroma"
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
                raise RuntimeError("openai SDK non disponibile")
            api_key = get_openai_key()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata (inseriscila nella sidebar o come env var)")
            self.client = OpenAI(api_key=api_key)
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers non disponibile")
            self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        print(f"[DEBUG] embed {len(texts)} testi con provider={self.provider_name} modello={self.model_name}")
        if self.use_openai:
            res = self.client.embeddings.create(model=self.model_name, input=texts)
            return [d.embedding for d in res.data]  # type: ignore
        return self.model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore


# ------------------------------
# VectorStore (Chroma wrapper)
# ------------------------------
class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        if chromadb is None:
            raise RuntimeError("chromadb non disponibile")
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)  # type: ignore
        )
        self.col = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, ids: List[str], documents: List[str], metadatas: List[dict], embeddings: List[List[float]]):
        print(f"[DEBUG] add {len(ids)} documenti a {self.collection_name}")
        self.col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_embeddings: List[List[float]], n_results: int = 5) -> dict:
        return self.col.query(query_embeddings=query_embeddings, n_results=n_results)

    def count(self) -> int:
        try:
            return self.col.count()  # type: ignore
        except Exception:
            return -1

def get_openai_key() -> Optional[str]:
    # override da UI se presente
    try:
        import streamlit as st  # type: ignore
        ui_key = st.session_state.get("openai_key")
        if ui_key:
            return ui_key
    except Exception:
        pass
    # fallback: environment variables già usate in app
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
                raise RuntimeError("openai SDK non disponibile")
            api_key = get_openai_key()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata (inseriscila nella sidebar o come env var)")
            self.client = OpenAI(api_key=api_key)
        elif provider == "Ollama (locale)":
            self.client = None  # REST semplice
        else:
            raise RuntimeError("Provider LLM non supportato")

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
        elif self.provider == "Ollama (locale)":
            import requests, json
            url = "http://localhost:11434/api/chat"
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
        return "Provider LLM non supportato"


RAG_SYSTEM_PROMPT = (
    "Sei un assistente tecnico che risponde basandosi su ticket YouTrack simili. "
    "Cita sempre gli ID dei ticket trovati tra parentesi quadre. Se il contesto non è sufficiente, chiedi chiarimenti."
)

def build_prompt(user_ticket: str, retrieved: List[Tuple[str, dict, float]]) -> str:
    print(f"[DEBUG] build_prompt con {len(retrieved)} documenti recuperati")
    parts = [
        "Nuovo ticket:\n" + user_ticket.strip(),
        "\nTicket simili trovati (dal più vicino):",
    ]
    for doc, meta, dist in retrieved:
        parts.append(
            f"- {meta.get('id_readable','')} | distanza={dist:.3f} | {meta.get('summary','')}\n{doc[:500]}"
        )
    parts.append("\nRispondi con citazioni tra [ ] degli ID dei ticket rilevanti.")
    return "\n".join(parts)


# ------------------------------
# NEW: utility per aprire la collection in sessione
# ------------------------------
def open_vector_in_session(persist_dir: str, collection_name: str):
    """Apre (o crea) una collection Chroma e la mette in sessione."""
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
# UI principale Streamlit
# ------------------------------
def run_streamlit_app():
    init_prefs_in_session()
    prefs = st.session_state.get("prefs", {})

    st.set_page_config(page_title="YouTrack RAG Support", layout="wide")
    st.title("YouTrack RAG Support")
    st.caption("Gestione ticket di supporto basata su storico YouTrack + RAG + LLM")

    with st.sidebar:
        st.header("Connessione YouTrack")
        yt_url = st.text_input("Server URL", value=prefs.get("yt_url", ""), placeholder="https://<org>.myjetbrains.com/youtrack")
        yt_token = st.text_input("Token (Bearer)", type="password")
        connect = st.button("Connetti YouTrack")


        st.markdown("---")
        st.header("Vector DB")
        persist_dir = st.text_input("Chroma path", value=prefs.get("persist_dir", DEFAULT_CHROMA_DIR), key="persist_dir")

        # Leggi le collections esistenti
        coll_options = []
        try:
            if chromadb is not None:
                _client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False)  # type: ignore
                )
                coll_options = [c.name for c in _client.list_collections()]  # type: ignore
        except Exception as e:
            st.caption(f"Impossibile leggere le collections da '{persist_dir}': {e}")

        NEW_LABEL = "➕ Crea nuova collection..."
        # Mantieni in sessione l'ultima scelta/nome nuova collection
        last_sel = st.session_state.get("collection_selected")
        last_new = st.session_state.get("new_collection_name", DEFAULT_COLLECTION)

        if coll_options:
            opts = coll_options + [NEW_LABEL]

            # Se esiste un'ultima selezione valida, usala; altrimenti predefinisci 0
            if last_sel in opts:
                default_index = opts.index(last_sel)
            elif DEFAULT_COLLECTION in opts:
                default_index = opts.index(DEFAULT_COLLECTION)
            else:
                default_index = 0

            sel = st.selectbox("Collection", options=opts, index=default_index, key="collection_select")

            if sel == NEW_LABEL:
                new_name = st.text_input("Nome nuova Collection", value=last_new, key="new_collection_name")
                collection_name = new_name.strip() or DEFAULT_COLLECTION
                # Registra che stiamo creando una nuova collection
                st.session_state["collection_selected"] = NEW_LABEL
            else:
                collection_name = sel
                st.session_state["collection_selected"] = sel
                st.session_state["new_collection_name"] = DEFAULT_COLLECTION  # reset opzionale
        else:
            st.caption("Nessuna collection trovata. Creane una nuova:")
            new_name = st.text_input("Nuova Collection", value=last_new, key="new_collection_name")
            collection_name = new_name.strip() or DEFAULT_COLLECTION
            st.session_state["collection_selected"] = NEW_LABEL

        # --- Gestione collection: elimina ---
        st.markdown("—")
        st.subheader("Gestione collection")
        is_existing_collection = collection_name in coll_options

        del_confirm = st.checkbox(
            f"Conferma eliminazione di '{collection_name}'",
            value=False,
            disabled=not is_existing_collection,
            help="Questa operazione rimuove definitivamente la collection dal datastore."
        )

        if st.button(
            "Elimina collection",
            type="secondary",
            disabled=not is_existing_collection,
            help="Rimuove definitivamente la collection selezionata dal vector datastore."
        ):
            if not del_confirm:
                st.warning("Spunta la casella di conferma per procedere con l'eliminazione.")
            else:
                try:
                    _client = chromadb.PersistentClient(
                        path=persist_dir,
                        settings=ChromaSettings(anonymized_telemetry=False)  # type: ignore
                    )
                    _client.delete_collection(name=collection_name)

                    # Rimuovi eventuale meta della collection (provider/modello)
                    meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
                    try:
                        if os.path.exists(meta_path):
                            os.remove(meta_path)
                    except Exception:
                        pass  # non bloccare l'UX se non riesci a cancellare il meta

                    # Pulisci stato se puntava alla collection rimossa
                    if st.session_state.get("vs_collection") == collection_name:
                        st.session_state["vector"] = None
                        st.session_state["vs_collection"] = None
                        st.session_state["vs_count"] = 0
                        st.session_state["vs_persist_dir"] = persist_dir  # opzionale: mantieni il path corrente

                    # Svuota eventuali risultati/issue caricati (opzionale ma consigliato)
                    st.session_state["issues"] = []

                    # Rimuovi selezione e nome nuova dalla sessione
                    st.session_state["collection_selected"] = None
                    st.session_state["new_collection_name"] = DEFAULT_COLLECTION

                    # Aggiorna anche le sticky prefs, se presenti
                    prefs = st.session_state.get("prefs", {})
                    prefs["collection_selected"] = None
                    prefs["new_collection_name"] = DEFAULT_COLLECTION
                    st.session_state["prefs"] = prefs
                    # Se hai definito save_prefs(), aggiorna il file su disco (safe no-op se non c'è)
                    try:
                        save_prefs(prefs)
                    except Exception:
                        pass

                    st.success(f"Collection '{collection_name}' eliminata con successo.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Errore durante l'eliminazione: {e}")


        st.markdown("---")
        # --- [SIDEBAR > Embeddings] sostituisci il blocco attuale con questo ---

        st.header("Embeddings")

        emb_provider_key = "emb_provider_select"
        emb_backend = st.selectbox(
            "Provider embeddings",
            ["Locale (sentence-transformers)", "OpenAI"],
            index=(0 if prefs.get("emb_backend", "Locale (sentence-transformers)") == "Locale (sentence-transformers)" else 1),
            key=emb_provider_key,
        )

        # Reset del modello se cambia il provider
        prev_emb_backend = st.session_state.get("last_emb_backend")
        if prev_emb_backend != emb_backend:
            st.session_state["last_emb_backend"] = emb_backend
            st.session_state["emb_model"] = "all-MiniLM-L6-v2" if emb_backend == "Locale (sentence-transformers)" else "text-embedding-3-small"

        # Default iniziale: prefs solo al primo giro
        if "emb_model" not in st.session_state:
            st.session_state["emb_model"] = prefs.get(
                "emb_model_name",
                "all-MiniLM-L6-v2" if emb_backend == "Locale (sentence-transformers)" else "text-embedding-3-small"
            )

        # Opzioni in base al provider
        emb_model_options = ["all-MiniLM-L6-v2"] if emb_backend == "Locale (sentence-transformers)" else ["text-embedding-3-small", "text-embedding-3-large"]

        # Se il valore corrente non è tra le opzioni (p.es. dopo switch), riallinea
        if st.session_state["emb_model"] not in emb_model_options:
            st.session_state["emb_model"] = emb_model_options[0]

        emb_model_name = st.selectbox(
            "Modello embeddings",
            options=emb_model_options,
            index=emb_model_options.index(st.session_state["emb_model"]),
            key="emb_model"
        )
        
        st.header("Retrieval")
        if "max_distance" not in st.session_state:
            st.session_state["max_distance"] = float(prefs.get("max_distance", 0.9))
        max_distance = st.slider("Soglia massima distanza (cosine)", 0.1, 2.0, st.session_state["max_distance"], 0.05)
        st.session_state["max_distance"] = max_distance

        st.markdown("---")
        st.header("LLM")
        llm_provider = st.selectbox(
            "Provider LLM",
            ["OpenAI", "Ollama (locale)"],
            index=(0 if prefs.get("llm_provider", "OpenAI") == "OpenAI" else 1),
            key="llm_provider_select",
        )

        # Se il provider cambia, resetta il modello al default del provider
        prev_provider = st.session_state.get("last_llm_provider")
        if prev_provider != llm_provider:
            st.session_state["last_llm_provider"] = llm_provider
            st.session_state["llm_model"] = (
                DEFAULT_LLM_MODEL if llm_provider == "OpenAI" else DEFAULT_OLLAMA_MODEL
            )

        # Default iniziale: prende dalle prefs SOLO al primo giro, poi governa lo stato
        llm_model_default = (DEFAULT_LLM_MODEL if llm_provider == "OpenAI" else DEFAULT_OLLAMA_MODEL)
        if "llm_model" not in st.session_state:
            st.session_state["llm_model"] = prefs.get("llm_model", llm_model_default)

        # Campo controllato via session_state: così il reset al cambio provider ha effetto
        llm_model = st.text_input("Modello LLM", key="llm_model")

        # Slider temperatura invariato (puoi lasciarlo com’è)
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
            help="Usata solo se scegli OpenAI come provider per Embeddings o LLM."
        )

        if openai_key_input:
            st.session_state["openai_key"] = openai_key_input

        if not openai_needed:
            st.caption("OpenAI API Key non necessaria: stai usando solo provider locali (Ollama / sentence-transformers).")
        elif (llm_provider == "Ollama (locale)") and (emb_backend == "OpenAI"):
            st.info("Stai usando: LLM = Ollama, Embeddings = OpenAI → la chiave verrà usata solo per gli embeddings.")

        st.markdown("---")
        st.header("Chat settings")
        top_k = st.slider("Numero risultati (k)", min_value=1, max_value=20, value=3, step=1)
        show_distances = st.checkbox("Mostra distanza nei risultati", value=True)

        st.header("Debug")
        if "show_prompt" not in st.session_state:
            st.session_state["show_prompt"] = bool(prefs.get("show_prompt", False))
        show_prompt = st.checkbox("Mostra prompt LLM", value=st.session_state["show_prompt"])
        st.session_state["show_prompt"] = show_prompt

        st.markdown("---")
        st.subheader("Preferenze")
        prefs_enabled = st.checkbox("Abilita memoria preferenze (locale)", value=True, help="Salva impostazioni non sensibili in .app_prefs.json")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Salva preferenze"):
                if prefs_enabled:
                    save_prefs({
                        "yt_url": yt_url,
                        "persist_dir": persist_dir,
                        "emb_backend": st.session_state.get("last_emb_backend", emb_backend),
                        "emb_model_name": st.session_state.get("emb_model"),
                        "llm_provider": st.session_state.get("last_llm_provider", llm_provider),
                        "llm_model": st.session_state.get("llm_model"),
                        "llm_temperature": st.session_state.get("llm_temperature", llm_temperature),
                        "max_distance": st.session_state.get("max_distance", max_distance),
                        "show_prompt": st.session_state.get("show_prompt", show_prompt),
                        "collection_selected": st.session_state.get("collection_selected"),
                        "new_collection_name": st.session_state.get("new_collection_name"),
                    })
                    st.session_state["prefs"] = load_prefs()
                    st.success("Preferenze salvate.")
                else:
                    st.info("Memoria preferenze disabilitata.")
        with c2:
            if st.button("Ripristina default"):
                try:
                    if os.path.exists(PREFS_PATH):
                        os.remove(PREFS_PATH)
                    st.session_state["prefs"] = {}
                    st.success("Preferenze ripristinate. Ricarica la pagina per vedere i default.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore nel ripristino: {e}")

        st.markdown("---")
        quit_btn = st.button("Quit")
        # Apertura automatica della collection selezionata (senza dover re-indicizzare)
        # Evita l'apertura se l'utente ha scelto "Crea nuova…" ma non ha digitato un nome diverso da default
        vector_ready = False
        if persist_dir and collection_name:
            changed = (
                st.session_state.get("vs_persist_dir") != persist_dir
                or st.session_state.get("vs_collection") != collection_name
                or st.session_state.get("vector") is None
            )

            # Non aprire se siamo su NEW_LABEL e il nome è vuoto (o solo default) e non esiste ancora
            if st.session_state.get("collection_selected") == NEW_LABEL and (collection_name == "" or (collection_name == DEFAULT_COLLECTION and collection_name not in coll_options)):
                pass  # aspetta che l'utente clicchi "Indicizza" per creare la nuova collection
            else:
                if changed:
                    ok, cnt, err = open_vector_in_session(persist_dir, collection_name)
                    vector_ready = ok
                    if ok:
                        st.caption(f"Collection '{collection_name}' aperta. Documenti indicizzati: {cnt if cnt>=0 else 'N/A'}")
                    else:
                        st.caption(f"Non riesco ad aprire la collection: {err}")
                else:
                    vector_ready = True
                    cnt = st.session_state.get("vs_count", -1)
                    st.caption(f"Collection '{collection_name}' pronta. Documenti indicizzati: {cnt if cnt>=0 else 'N/A'}")

        if quit_btn:
            st.write("Chiusura applicazione...")
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
            st.error("Inserisci URL e Token di YouTrack")
        else:
            try:
                st.session_state["yt_client"] = YouTrackClient(yt_url, yt_token)
                st.success("Connesso a YouTrack")
                with st.spinner("Carico progetti..."):
                    st.session_state["projects"] = st.session_state["yt_client"].list_projects()
            except Exception as e:
                st.error(f"Connessione fallita: {e}")

    # Projects & issues
    # --- aggiungi questo import assieme agli altri in cima al file ---

    st.subheader("Progetti YouTrack")
    if st.session_state.get("yt_client"):
        projects = st.session_state.get("projects", [])
        if projects:
            names = [f"{p.get('name')} ({p.get('shortName')})" for p in projects]
            sel = st.selectbox("Scegli progetto", options=["-- seleziona --"] + names, key="proj_select")

            if sel and sel != "-- seleziona --":
                p = projects[names.index(sel)]
                project_key = p.get("shortName") or p.get("name")

                # AUTO-LOAD: quando cambi progetto carico subito gli issue
                prev_key = st.session_state.get("last_project_key")
                if prev_key != project_key:
                    try:
                        with st.spinner("Carico issues..."):
                            st.session_state["issues"] = st.session_state["yt_client"].list_issues(project_key)
                        st.session_state["last_project_key"] = project_key
                        st.success(f"Caricati {len(st.session_state['issues'])} issues")
                    except Exception as e:
                        st.error(f"Errore caricamento issues: {e}")

                # opzionale: pulsante per ricaricare manualmente
                if st.button("Ricarica issues"):
                    try:
                        with st.spinner("Ricarico issues..."):
                            st.session_state["issues"] = st.session_state["yt_client"].list_issues(project_key)
                        st.success(f"Caricati {len(st.session_state['issues'])} issues")
                    except Exception as e:
                        st.error(f"Errore ricaricamento: {e}")
        else:
            st.caption("Nessun progetto trovato (o permessi insufficienti).")

    # Mostra SEMPRE la tabella se ci sono issue in memoria
    # --- sostituisci il blocco che mostra la tabella degli issue con questo ---
    issues = st.session_state.get("issues", [])
    if issues:
        base_url = (st.session_state.get("yt_client").base_url if st.session_state.get("yt_client") else "").rstrip("/")

        # Costruisci una tabella Markdown con ID cliccabile e senza Project/Apri
        lines = []
        lines.append("| ID | Summary |")
        lines.append("|---|---|")
        for it in issues:
            url = f"{base_url}/issue/{it.id_readable}" if base_url else ""
            id_cell = f"[{it.id_readable}]({url})" if url else it.id_readable
            # taglia summary lunghissime (facoltativo)
            summary = it.summary.strip().replace("\n", " ")
            if len(summary) > 160:
                summary = summary[:157] + "..."
            lines.append(f"| {id_cell} | {summary} |")

        st.markdown("\n".join(lines), unsafe_allow_html=False)
    else:
        st.caption("Nessun issue in memoria. Seleziona un progetto per caricare i ticket.")

    # Ingest
    st.subheader("Indicizzazione Vector DB")
    col1, col2 = st.columns([1, 3])
    with col1:
        start_ingest = st.button("Indicizza ticket")
    with col2:
        st.caption("Crea embeddings dei ticket e li salva su Chroma per il retrieval semantico")

    if start_ingest:
        issues = st.session_state.get("issues", [])
        if not issues:
            st.error("Prima carica i ticket del progetto")
        else:
            try:
                st.session_state["vector"] = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
                use_openai_embeddings = (emb_backend == "OpenAI")
                st.session_state["embedder"] = EmbeddingBackend(use_openai=use_openai_embeddings, model_name=emb_model_name)
                # Salva meta modello embeddings per coerenza
                try:
                    meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
                    meta = {"provider": st.session_state["embedder"].provider_name, "model": st.session_state["embedder"].model_name}
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f)
                except Exception:
                    pass

                ids = [it.id_readable for it in issues]
                docs = [it.text_blob() for it in issues]
                metas = [{"id_readable": it.id_readable, "summary": it.summary, "project": it.project} for it in issues]
                with st.spinner("Calcolo embeddings e salvo su Chroma..."):
                    embs = st.session_state["embedder"].embed(
                        [f"{m['id_readable']} | {m['summary']}\n\n{d}" for d, m in zip(docs, metas)]
                    )
                    st.session_state["vector"].add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                # aggiorna contatore
                st.session_state["vs_persist_dir"] = persist_dir
                st.session_state["vs_collection"] = collection_name
                st.session_state["vs_count"] = st.session_state["vector"].count()
                st.success(f"Indicizzazione completata. Totale documenti: {st.session_state['vs_count']}")
            except Exception as e:
                st.error(f"Errore indicizzazione: {e}")

    # Chat
    st.subheader("Chatbot RAG")
    query = st.text_area("Nuovo ticket", height=140, placeholder="Descrivi il problema come faresti aprendo un ticket")
    run_chat = st.button("Cerca e rispondi")

    if run_chat:
        if not query.strip():
            st.error("Inserisci il testo del ticket")
        elif not st.session_state.get("vector"):
            ok, _, _ = open_vector_in_session(persist_dir, collection_name)
            if not ok:
                st.error("Apri o crea una collection valida nella sidebar")
        else:
            try:
                if st.session_state.get("embedder") is None:
                    use_openai_embeddings = (emb_backend == "OpenAI")
                    st.session_state["embedder"] = EmbeddingBackend(
                        use_openai=use_openai_embeddings,
                        model_name=emb_model_name
                    )

                # Avvisa se il modello corrente differisce da quello usato per l'indicizzazione
                try:
                    meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        if (m.get("provider") != st.session_state["embedder"].provider_name) or \
                        (m.get("model") != st.session_state["embedder"].model_name):
                            st.info(
                                f"Nota: la collection è stata indicizzata con {m.get('provider')} / {m.get('model')}; "
                                f"stai cercando con {st.session_state['embedder'].provider_name} / "
                                f"{st.session_state['embedder'].model_name}."
                            )
                except Exception:
                    pass

                # Retrieval
                q_emb = st.session_state["embedder"].embed([query])
                res = st.session_state["vector"].query(query_embeddings=q_emb, n_results=top_k)
                docs  = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]

                # Filtra risultati per distanza
                DIST_MAX = max_distance
                retrieved = [
                    (doc, meta, dist)
                    for doc, meta, dist in zip(docs, metas, dists)
                    if dist is not None and dist <= DIST_MAX
                ]

                # Prompt
                if retrieved:
                    prompt = build_prompt(query, retrieved)
                else:
                    prompt = f"Nuovo ticket:\n{query.strip()}\n\nNessun ticket simile è stato trovato nel knowledge base."

                if st.session_state.get("show_prompt", False):
                    with st.expander("Prompt inviato al LLM", expanded=False):
                        st.code(prompt, language="markdown")

                # LLM
                llm = LLMBackend(llm_provider, llm_model, temperature=llm_temperature)
                with st.spinner("Genero risposta..."):
                    answer = llm.generate(RAG_SYSTEM_PROMPT, prompt)
                st.write(answer)

                # Mostra risultati simili una sola volta (cliccabili) SOLO se esistono match sotto soglia
                if retrieved:
                    base_url = (st.session_state.get("yt_client").base_url if st.session_state.get("yt_client") else "").rstrip("/")
                    st.write("Risultati simili (top-k):")
                    for (doc, meta, dist) in retrieved:
                        idr = meta.get("id_readable", "")
                        summary = meta.get("summary", "")
                        url = f"{base_url}/issue/{idr}" if base_url and idr else ""
                        if url:
                            if show_distances:
                                st.markdown(f"- [{idr}]({url}) — distanza={dist:.3f} — {summary}")
                            else:
                                st.markdown(f"- [{idr}]({url}) — {summary}")
                        else:
                            if show_distances:
                                st.write(f"- {idr} — distanza={dist:.3f} — {summary}")
                            else:
                                st.write(f"- {idr} — {summary}")
                else:
                    st.caption("Nessun risultato sufficientemente simile (sotto soglia).")

            except Exception as e:
                st.error(f"Errore chat: {e}")


# ------------------------------
# CLI + Self-tests (opzionali)
# ------------------------------
def _cli_help():
    print("Uso: streamlit run app.py --server.port 8502")

def _self_tests():
    print("Eseguo self-tests minimi...")
    vs = None
    try:
        vs = VectorStore(DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION)
        print("VectorStore OK.")
    except Exception as e:
        print(f"VectorStore non disponibile: {e}")

    try:
        emb = EmbeddingBackend(use_openai=False, model_name=DEFAULT_EMBEDDING_MODEL)
        vec = emb.embed(["testo uno", "testo due"])
        assert len(vec) == 2 and isinstance(vec[0], list)
        print("Embeddings locali OK.")
    except Exception as e:
        print(f"Embeddings locali non disponibili: {e}")

    try:
        llm = LLMBackend("OpenAI", DEFAULT_LLM_MODEL)
        assert isinstance(llm, LLMBackend)
        print("LLM OpenAI OK (init).")
    except Exception as e:
        print(f"LLM OpenAI non disponibile: {e}")

    try:
        llm = LLMBackend("Ollama (locale)", DEFAULT_OLLAMA_MODEL)
        assert isinstance(llm, LLMBackend)
        print("LLM Ollama OK (init).")
    except Exception as e:
        print(f"LLM Ollama non disponibile: {e}")

    try:
        llm = LLMBackend("???", "x")
    except Exception as e:
        assert "Provider LLM non supportato" in str(e)
    print("Tutti i self-tests sono PASSATI. ✅")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    print("[DEBUG] Avvio programma principale.")
    if ST_AVAILABLE:
        print("[DEBUG] Avvio interfaccia Streamlit.")
        run_streamlit_app()
    else:
        _cli_help()
        _self_tests()
