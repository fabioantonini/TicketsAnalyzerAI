"""
Streamlit RAG Support App (completa, con debug + self-tests + UI + Quit)

Funzionalità
1) Connessione a YouTrack (URL + token) e lista progetti/issues
2) Ingestione dei ticket in un Vector DB locale (Chroma) con embeddings locali o OpenAI
3) Selezione LLM (OpenAI o locale via Ollama)
4) Chat RAG: ricerca ticket simili + risposta LLM con citazioni ID
5) Modalità CLI con self-tests quando Streamlit non è disponibile
6) Pulsante Quit nella sidebar per chiudere l’app dal browser

Avvio consigliato
streamlit run app.py --server.port 8502
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd

# ------------------------------
# Streamlit (opzionale per CLI)
# ------------------------------
try:
    import streamlit as st  # type: ignore
    ST_AVAILABLE = True
    print("[DEBUG] Streamlit importato correttamente.")
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

# ------------------------------
# Vector store / embeddings
# ------------------------------
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
DEFAULT_CHROMA_DIR = os.path.join("data", "chroma")
DEFAULT_COLLECTION = "youtrack_tickets"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------------------------------
# Data model
# ------------------------------
@dataclass
class YTIssue:
    id_readable: str
    summary: str
    description: str
    project: str

    def text_blob(self) -> str:
        desc = self.description or ""
        return f"[{self.id_readable}] {self.summary}\n\n{desc}"

# ------------------------------
# YouTrack client
# ------------------------------
class YouTrackClient:
    def __init__(self, base_url: str, token: str):
        print(f"[DEBUG] Inizializzazione YouTrackClient con base_url={base_url}")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        print(f"[DEBUG] GET {url} con params={params}")
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_projects(self) -> List[Dict[str, Any]]:
        print("[DEBUG] list_projects chiamato")
        fields = "id,name,shortName"
        return self._get("/api/admin/projects", params={"fields": fields})

    def list_issues(self, project_key: str, limit: int = 200) -> List[YTIssue]:
        print(f"[DEBUG] list_issues per progetto={project_key}")
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
        print(f"[DEBUG] {len(issues)} ticket recuperati.")
        return issues

# ------------------------------
# Embeddings + VectorStore
# ------------------------------
class EmbeddingBackend:
    def __init__(self, use_openai: bool, model_name: str = DEFAULT_EMBEDDING_MODEL):
        print(f"[DEBUG] Inizializzazione EmbeddingBackend, use_openai={use_openai}, model={model_name}")
        self.use_openai = use_openai
        self.model_name = model_name
        self.client = None
        self.local_model = None
        if use_openai:
            if OpenAI is None:
                raise RuntimeError("openai SDK non installato")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata")
            self.client = OpenAI(api_key=api_key)
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers non installato")
            self.local_model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        print(f"[DEBUG] embed chiamato su {len(texts)} testi")
        if self.use_openai:
            try:
                res = self.client.embeddings.create(model="text-embedding-3-small", input=texts)  # type: ignore
                return [d.embedding for d in res.data]  # type: ignore
            except Exception as e:
                print(f"[DEBUG] Errore embeddings OpenAI: {e}")
                raise
        else:
            return self.local_model.encode(texts, convert_to_numpy=False, show_progress_bar=False).tolist()  # type: ignore

class VectorStore:
    def __init__(self, persist_dir: str = DEFAULT_CHROMA_DIR, collection_name: str = DEFAULT_COLLECTION):
        print(f"[DEBUG] Inizializzazione VectorStore dir={persist_dir}, collection={collection_name}")
        if chromadb is None:
            raise RuntimeError("chromadb non installato")
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))  # type: ignore
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids: List[str], texts: List[str], metadatas: List[dict], embeddings: Optional[List[List[float]]] = None):
        print(f"[DEBUG] Aggiungo {len(ids)} documenti a VectorStore")
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)  # type: ignore

    def query(self, query_embeddings: List[List[float]], n_results: int = 5) -> Dict[str, Any]:
        print(f"[DEBUG] Query VectorStore con n_results={n_results}")
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results, include=["documents", "metadatas", "distances"])  # type: ignore

# ------------------------------
# LLM backend
# ------------------------------
class LLMBackend:
    def __init__(self, provider: str, model: str, temperature: float = 0.2):
        print(f"[DEBUG] Inizializzazione LLMBackend provider={provider}, model={model}")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.openai_client = None
        if provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai SDK non installato")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata")
            self.openai_client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        print(f"[DEBUG] generate chiamato con provider={self.provider}")
        if self.provider == "openai":
            try:
                rsp = self.openai_client.responses.create(  # type: ignore
                    model=self.model,
                    temperature=self.temperature,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                if getattr(rsp, "output_text", None):  # type: ignore
                    return rsp.output_text  # type: ignore
                chunks = []
                for item in rsp.output:  # type: ignore
                    if item.type == "message":
                        for ct in item.message.content:  # type: ignore
                            if getattr(ct, "type", "") == "output_text":
                                chunks.append(ct.text)
                return "".join(chunks) if chunks else ""
            except Exception as e:
                print(f"[DEBUG] Fallback chat.completions per errore: {e}")
                try:
                    chat = self.openai_client.chat.completions.create(  # type: ignore
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    return chat.choices[0].message.content  # type: ignore
                except Exception as e2:
                    print(f"[DEBUG] Errore OpenAI legacy: {e2}")
                    return f"Errore OpenAI: {e2}"
        elif self.provider == "ollama":
            try:
                payload = {
                    "model": self.model,
                    "prompt": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}",
                    "options": {"temperature": self.temperature},
                    "stream": False,
                }
                print("[DEBUG] Chiamo Ollama API locale")
                r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")
            except Exception as e:
                print(f"[DEBUG] Errore Ollama: {e}")
                return f"Errore Ollama: {e}"
        else:
            print("[DEBUG] Provider LLM non supportato")
            return "Provider LLM non supportato"

# ------------------------------
# Prompting
# ------------------------------
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
    parts.append("\nIstruzioni: proponi una risposta sintetica e operativa. Includi i riferimenti ai ticket fra [].")
    return "\n\n".join(parts)

# ------------------------------
# UI Streamlit
# ------------------------------

def run_streamlit_app() -> None:
    st.set_page_config(page_title="Support RAG for YouTrack", page_icon="🧭", layout="wide")

    def _err(msg: str):
        try:
            st.toast(msg, icon="⚠️")
        except Exception:
            st.warning(msg)

    st.title("Support RAG per YouTrack")
    st.caption("Gestione ticket di supporto basata su storico YouTrack + RAG + LLM")

    with st.sidebar:
        st.header("Connessione YouTrack")
        yt_url = st.text_input("Server URL", placeholder="https://<org>.myjetbrains.com/youtrack")
        yt_token = st.text_input("Token (Bearer)", type="password")
        connect = st.button("Connetti YouTrack")

        st.markdown("---")
        st.header("Vector DB")
        persist_dir = st.text_input("Chroma path", value=DEFAULT_CHROMA_DIR)
        collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)

        st.markdown("---")
        st.header("Embeddings")
        emb_backend = st.selectbox("Provider embeddings", ["Locale (sentence-transformers)", "OpenAI"], index=0)
        emb_model_name = st.text_input("Modello embeddings", value=DEFAULT_EMBEDDING_MODEL)

        st.markdown("---")
        st.header("LLM")
        llm_provider = st.selectbox("Provider LLM", ["OpenAI", "Ollama (locale)"])
        llm_model = st.text_input("Modello LLM", value=("gpt-4o-mini" if llm_provider == "OpenAI" else DEFAULT_OLLAMA_MODEL))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

        st.markdown("---")
        quit_btn = st.button("Quit")
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
            _err("Inserisci URL e Token di YouTrack")
        else:
            try:
                st.session_state["yt_client"] = YouTrackClient(yt_url, yt_token)
                st.success("Connesso a YouTrack")
                with st.spinner("Carico progetti..."):
                    st.session_state["projects"] = st.session_state["yt_client"].list_projects()
            except Exception as e:
                _err(f"Connessione fallita: {e}")

    # Projects & issues
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Progetti")
        if st.session_state["projects"]:
            proj_options = {f"{p.get('shortName') or p.get('name')}": (p.get("shortName") or p.get("name")) for p in st.session_state["projects"]}
            proj_label = st.selectbox("Seleziona progetto", list(proj_options.keys()))
            load_issues = st.button("Carica ticket")
        else:
            st.info("Connetti YouTrack per elencare i progetti")
            proj_label = None
            load_issues = False

    with right:
        st.subheader("Ticket progetto")
        if load_issues and proj_label:
            key = proj_options[proj_label]  # type: ignore
            try:
                with st.spinner("Scarico ticket..."):
                    issues = st.session_state["yt_client"].list_issues(project_key=key, limit=500)
                    st.session_state["issues"] = issues
            except Exception as e:
                _err(f"Errore caricamento ticket: {e}")

        if st.session_state["issues"]:
            df = pd.DataFrame([
                {
                    "id": it.id_readable,
                    "summary": it.summary,
                    "description": (it.description or "").strip().replace("\n", " ")[:2000],
                    "project": it.project,
                }
                for it in st.session_state["issues"]
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("Nessun ticket caricato")

    # Ingestion
    st.markdown("")
    st.subheader("Indicizzazione Vector DB")
    col1, col2 = st.columns([1, 3])
    with col1:
        start_ingest = st.button("Indicizza ticket")
    with col2:
        st.caption("Crea embeddings dei ticket e li salva su Chroma per il retrieval semantico")

    if start_ingest:
        issues = st.session_state.get("issues", [])
        if not issues:
            _err("Prima carica i ticket del progetto")
        else:
            try:
                st.session_state["vector"] = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
                use_openai_embeddings = (emb_backend == "OpenAI")
                st.session_state["embedder"] = EmbeddingBackend(use_openai=use_openai_embeddings, model_name=emb_model_name)
                ids = [it.id_readable for it in issues]
                docs = [it.text_blob() for it in issues]
                metas = [{"id_readable": it.id_readable, "summary": it.summary, "project": it.project} for it in issues]
                with st.spinner("Calcolo embeddings e salvo su Chroma..."):
                    embs = st.session_state["embedder"].embed(docs)
                    st.session_state["vector"].add(ids=ids, texts=docs, metadatas=metas, embeddings=embs)
                st.success(f"Indicizzati {len(ids)} ticket nella collection '{collection_name}'")
            except Exception as e:
                _err(f"Errore indicizzazione: {e}")

    # Chatbot
    st.markdown("")
    st.subheader("Chatbot RAG")
    st.caption("Inserisci il testo di un nuovo ticket: il sistema cercherà casi simili e proporrà una risposta")

    query = st.text_area("Nuovo ticket", height=140, placeholder="Descrivi il problema come faresti aprendo un ticket")
    run_chat = st.button("Cerca e rispondi")

    if run_chat:
        if not query.strip():
            _err("Inserisci il testo del ticket")
        elif not st.session_state.get("vector"):
            _err("Indicizza i ticket prima di usare il chatbot")
        else:
            try:
                if st.session_state.get("embedder") is None:
                    use_openai_embeddings = (emb_backend == "OpenAI")
                    st.session_state["embedder"] = EmbeddingBackend(use_openai=use_openai_embeddings, model_name=emb_model_name)

                q_emb = st.session_state["embedder"].embed([query])
                res = st.session_state["vector"].query(query_embeddings=q_emb, n_results=5)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                retrieved = list(zip(docs, metas, dists))

                provider_key = "openai" if llm_provider.lower().startswith("openai") else "ollama"
                llm = LLMBackend(provider=provider_key, model=llm_model, temperature=temperature)

                prompt = build_prompt(query, retrieved)
                with st.spinner("Genero risposta LLM..."):
                    answer = llm.generate(RAG_SYSTEM_PROMPT, prompt)

                st.markdown("Risposta")
                st.write(answer)

                with st.expander("Ticket simili usati come contesto"):
                    for (doc, meta, dist) in retrieved:
                        st.write(f"[{meta.get('id_readable','')}] {meta.get('summary','')} | distanza={dist:.3f}")
                        st.code(doc[:1200])

            except Exception as e:
                _err(f"Errore chatbot: {e}")

    # Footer
    st.markdown("---")
    st.caption("Suggerimenti: usa embedding locali per privacy; programma un job periodico per sincronizzare i ticket; aggiungi rating delle risposte per migliorare i prompt.")

# ------------------------------
# CLI help + Self tests (senza Streamlit)
# ------------------------------

def _cli_help() -> None:
    print("\n=== Support RAG for YouTrack (CLI) ===")
    print("Streamlit non è installato nell'ambiente corrente.")
    print("Installa i requisiti e avvia l'app web:")
    print("  pip install -U streamlit requests chromadb sentence-transformers openai tiktoken pandas")
    print("  # opzionale per modelli locali: pip install -U ollama")
    print("  streamlit run app.py\n")


def _self_tests() -> None:
    print("Eseguo self-tests...")

    yt = YouTrackClient("https://example.youtrack", "XYZ")
    assert yt.base_url == "https://example.youtrack"
    assert yt.session.headers["Authorization"].startswith("Bearer ")

    retrieved = [
        ("[YT-1] Wifi unstable\n\nDetails...", {"id_readable": "YT-1", "summary": "Wifi unstable"}, 0.1234),
        ("[YT-2] Router reboot\n\nDetails...", {"id_readable": "YT-2", "summary": "Router reboot"}, 0.9876),
    ]
    p = build_prompt("Problema con la rete", retrieved)
    assert "Nuovo ticket:" in p and "Ticket simili" in p
    assert "- YT-1 | distanza=0.123 | Wifi unstable" in p
    assert "- YT-2 | distanza=0.988 | Router reboot" in p

    llm = LLMBackend(provider="other", model="x")
    assert llm.generate("s", "u") == "Provider LLM non supportato"

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
