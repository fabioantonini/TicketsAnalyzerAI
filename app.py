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
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_EXPERIMENTS")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata")
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


# ------------------------------
# LLM Backend
# ------------------------------
class LLMBackend:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        if provider == "OpenAI":
            if OpenAI is None:
                raise RuntimeError("openai SDK non disponibile")
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_EXPERIMENTS")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata")
            self.client = OpenAI(api_key=api_key)
        elif provider == "Ollama (locale)":
            self.client = None  # usi REST semplice
        else:
            raise RuntimeError("Provider LLM non supportato")

    def generate(self, system: str, user: str) -> str:
        if self.provider == "OpenAI":
            try:
                # Responses API
                res = self.client.responses.create(
                    model=self.model_name,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return res.output_text  # type: ignore
            except Exception:
                # Fallback Chat Completions
                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return chat.choices[0].message.content or ""
        elif self.provider == "Ollama (locale)":
            import requests, json

            url = "http://localhost:11434/api/chat"
            payload = {
                "model": self.model_name,  # es: "llama3.2" o "llama3.2:latest"
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False  # fondamentale per evitare JSON concatenati
            }

            r = requests.post(url, json=payload, timeout=300)
            try:
                r.raise_for_status()
                data = r.json()  # ora deve essere un singolo JSON
            except json.JSONDecodeError:
                # Fallback: se arriva ancora streaming, ricomponi i chunk JSON
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
                        elif "response" in obj:  # compat con /api/generate
                            parts.append(obj["response"])
                return "\n".join(parts).strip() if parts else text

            # Risposta non-stream (singolo JSON)
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                    return data["message"]["content"]
                if "response" in data:  # compat con /api/generate
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
    st.set_page_config(page_title="YouTrack RAG Support", layout="wide")
    st.title("YouTrack RAG Support")
    st.caption("Gestione ticket di supporto basata su storico YouTrack + RAG + LLM")

    with st.sidebar:
        st.header("Connessione YouTrack")
        yt_url = st.text_input("Server URL", placeholder="https://<org>.myjetbrains.com/youtrack")
        yt_token = st.text_input("Token (Bearer)", type="password")
        connect = st.button("Connetti YouTrack")

        st.markdown("---")
        st.header("Vector DB")
        persist_dir = st.text_input("Chroma path", value=DEFAULT_CHROMA_DIR)
        # Legge le collections esistenti e permette di crearne una nuova
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
        if coll_options:
            opts = coll_options + [NEW_LABEL]
            default_index = opts.index(DEFAULT_COLLECTION) if DEFAULT_COLLECTION in opts else 0
            sel = st.selectbox("Collection", options=opts, index=default_index)
            collection_name = st.text_input("Nome nuova Collection", value=DEFAULT_COLLECTION) if sel == NEW_LABEL else sel
        else:
            st.caption("Nessuna collection trovata. Creane una nuova:")
            collection_name = st.text_input("Nuova Collection", value=DEFAULT_COLLECTION)

        st.markdown("---")
        # --- Sostituisci il blocco "Embeddings" con questo ---
        st.header("Embeddings")
        emb_backend = st.selectbox("Provider embeddings", ["Locale (sentence-transformers)", "OpenAI"], index=0)

        # Popola i modelli in base al provider scelto
        if emb_backend == "Locale (sentence-transformers)":
            emb_model_options = ["all-MiniLM-L6-v2"]
        else:  # OpenAI
            emb_model_options = ["text-embedding-3-small", "text-embedding-3-large"]

        emb_model_name = st.selectbox("Modello embeddings", options=emb_model_options, index=0, key="emb_model_select")

        st.markdown("---")
        st.header("LLM")
        llm_provider = st.selectbox("Provider LLM", ["OpenAI", "Ollama (locale)"])
        llm_model = st.text_input("Modello LLM", value=DEFAULT_LLM_MODEL if llm_provider == "OpenAI" else DEFAULT_OLLAMA_MODEL)

        st.markdown("---")
        quit_btn = st.button("Quit")
        # Apertura automatica della collection selezionata (senza dover re-indicizzare)
        vector_ready = False
        if persist_dir and collection_name:
            changed = (
                st.session_state.get("vs_persist_dir") != persist_dir
                or st.session_state.get("vs_collection") != collection_name
                or st.session_state.get("vector") is None
            )
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
    import pandas as pd

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
    issues = st.session_state.get("issues", [])
    if issues:
        base_url = (st.session_state.get("yt_client").base_url if st.session_state.get("yt_client") else "").rstrip("/")
        rows = []
        for it in issues:
            url = f"{base_url}/issue/{it.id_readable}" if base_url else ""
            rows.append({
                "ID": it.id_readable,
                "Summary": it.summary,
                "Project": it.project,
                "URL": url
            })
        df = pd.DataFrame(rows, columns=["ID", "Summary", "Project", "URL"])
        st.dataframe(df, use_container_width=True)
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
                    st.session_state["embedder"] = EmbeddingBackend(use_openai=use_openai_embeddings, model_name=emb_model_name)

                # Avvisa se il modello corrente differisce da quello usato per l'indicizzazione
                try:
                    meta_path = os.path.join(persist_dir, f"{collection_name}__meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        if (m.get("provider") != st.session_state["embedder"].provider_name) or (m.get("model") != st.session_state["embedder"].model_name):
                            st.info(f"Nota: la collection è stata indicizzata con {m.get('provider')} / {m.get('model')}; stai cercando con {st.session_state['embedder'].provider_name} / {st.session_state['embedder'].model_name}.")
                except Exception:
                    pass

                q_emb = st.session_state["embedder"].embed([query])
                res = st.session_state["vector"].query(query_embeddings=q_emb, n_results=5)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                retrieved = list(zip(docs, metas, dists))
                prompt = build_prompt(query, retrieved)
                llm = LLMBackend(llm_provider, llm_model)
                with st.spinner("Genero risposta..."):
                    answer = llm.generate(RAG_SYSTEM_PROMPT, prompt)
                st.write(answer)

                # Tabella risultati
                st.write("Risultati simili (top-k):")
                for (doc, meta, dist) in retrieved:
                    idr = meta.get("id_readable", "")
                    summary = meta.get("summary", "")
                    st.write(f"- [{idr}] distanza={dist:.3f} — {summary}")

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
