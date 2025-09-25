#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit RAG Support App + Governance Monitor (AI)

Superset dell'app precedente: mantiene tutte le feature RAG originali
(progetti, issues, Chroma, embeddings locali/OpenAI, chatbot, Quit,
self-tests) e aggiunge il monitor AI per il progetto di Governance
(analisi scadenze, flag Overdue/DelayRisk, commento AI, AutoFlag).

Avvio consigliato
streamlit run app_gov.py --server.port 8502
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

import requests
import pandas as pd

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

DEFAULT_CHROMA_DIR = os.path.join("data", "chroma")
DEFAULT_COLLECTION = "youtrack_tickets"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------------------------------
# YouTrack client e modelli
# ------------------------------

@dataclass
class YTIssue:
    id_readable: str
    summary: str
    description: str
    project: str

    def text_blob(self) -> str:
        desc = self.description or ""
        return f"[{self.id_readable}] {self.summary} {desc}"


@dataclass
class YTConfig:
    base_url: str
    token: str


class YouTrackClient:
    def __init__(self, base_url: str, token: str):
        print(f"[DEBUG] Inizializzazione YouTrackClient con base_url={base_url}")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        print(f"[DEBUG] GET {url} con params={params}")
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        print(f"[DEBUG] POST {url}")
        r = self.session.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json() if r.text else {}

    # Originali
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

    # Estensioni Governance
    def list_issues_governance(self, project_key: str, limit: int = 500) -> List[Dict[str, Any]]:
        fields = (
            "id,idReadable,summary,project(shortName),dueDate,created,updated,"
            "customFields(name,value(name,value))"
        )
        params = {
            "query": f"project: {project_key}",
            "$top": str(limit),
            "fields": fields,
        }
        data = self._get("/api/issues", params=params)
        issues: List[Dict[str, Any]] = []
        for it in data:
            cf_map = {cf.get("name"): cf.get("value") for cf in it.get("customFields", [])}

            def _enum_name(v):
                return (v or {}).get("name") if isinstance(v, dict) else v

            issues.append({
                "id": it.get("id"),
                "id_readable": it.get("idReadable"),
                "summary": it.get("summary", ""),
                "project": (it.get("project") or {}).get("shortName", ""),
                "type": _enum_name(cf_map.get("Type")),
                "owner_team": _enum_name(cf_map.get("Owner Team")),
                "autoflag": _enum_name(cf_map.get("AutoFlag")),
                "blocked": _enum_name(cf_map.get("Blocked")),
                "confidence": cf_map.get("Confidence"),
                "due_ms": it.get("dueDate"),
                "sla_ms": cf_map.get("SLA Target"),
                "created": it.get("created"),
                "updated": it.get("updated"),
            })
        return issues

    def add_comment(self, issue_id: str, text: str) -> None:
        self._post(f"/api/issues/{issue_id}/comments?fields=id", {"text": text})

    def set_autoflag(self, issue_id: str, flag: Optional[str]) -> None:
        payload = {
            "customFields": [
                {
                    "$type": "SingleEnumIssueCustomField",
                    "name": "AutoFlag",
                    "value": {"name": flag} if flag else None,
                }
            ]
        }
        self._post(f"/api/issues/{issue_id}?fields=id", payload)


# ------------------------------
# Embeddings, Vector store, LLM
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
            api_key = os.getenv("OPENAI_API_KEY_EXPERIMENTS") or os.getenv("OPENAI_API_KEY")
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
            res = self.client.embeddings.create(model="text-embedding-3-small", input=texts)  # type: ignore
            return [d.embedding for d in res.data]  # type: ignore
        else:
            embs = self.local_model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return embs.tolist()


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

    def count(self) -> int:
        try:
            return self.collection.count()  # type: ignore
        except Exception:
            return -1


class LLMBackend:
    def __init__(self, provider: str, model: str, temperature: float = 0.2):
        print(f"[DEBUG] Inizializzazione LLMBackend provider={provider}, model={model}")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.openai_client = None
        if provider.lower() == "openai":
            if OpenAI is None:
                raise RuntimeError("openai SDK non installato")
            api_key = os.getenv("OPENAI_API_KEY_EXPERIMENTS") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata")
            self.openai_client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        print(f"[DEBUG] generate chiamato con provider={self.provider}")
        if self.provider.lower() == "openai":
            try:
                chat = self.openai_client.chat.completions.create(  # type: ignore
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return chat.choices[0].message.content or ""  # type: ignore
            except Exception as e2:
                print(f"[DEBUG] Errore OpenAI: {e2}")
                return f"Errore OpenAI: {e2}"
        elif self.provider.lower() == "ollama":
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


RAG_SYSTEM_PROMPT = (
    "Sei un assistente tecnico che risponde basandosi su ticket YouTrack simili. "
    "Cita sempre gli ID dei ticket trovati tra parentesi quadre. Se il contesto non è sufficiente, chiedi chiarimenti."
)

# ------------------------------
# Prompt helper
# ------------------------------

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
# Utilità date per Governance
# ------------------------------

def _ms_to_date(ms: Optional[int]) -> Optional[datetime]:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _date_to_str(d: Optional[datetime]) -> str:
    return d.strftime("%Y-%m-%d") if d else ""


def compute_flag(issue: Dict[str, Any]) -> str:
    due = _ms_to_date(issue.get("due_ms"))
    if not due:
        return "None"
    today = datetime.now(timezone.utc)
    if due.date() < today.date():
        return "Overdue"
    days = (due.date() - today.date()).days
    if days <= 3:
        return "DelayRisk"
    return "None"


def ai_assess_actions(llm: LLMBackend, issue: Dict[str, Any]) -> str:
    sys_prompt = (
        "Sei un PMO AI. Fornisci un breve stato (1 riga) e 3-5 azioni pratiche. "
        "Chiudi con un ETA proposto in giorni."
    )
    user = (
        f"Ticket: [{issue.get('id_readable')}] {issue.get('summary')}\n"
        f"Type={issue.get('type')} Team={issue.get('owner_team')} "
        f"AutoFlag={issue.get('autoflag')} Blocked={issue.get('blocked')} "
        f"Confidence={issue.get('confidence')} Due={_date_to_str(_ms_to_date(issue.get('due_ms')))} "
        f"SLA={_date_to_str(_ms_to_date(issue.get('sla_ms')))}\n"
        "Obiettivo: suggerisci azioni rapide e priorità."
    )
    return llm.generate(sys_prompt, user)

# ------------------------------
# App Streamlit
# ------------------------------

def run_streamlit_app() -> None:
    st.set_page_config(page_title="Support RAG + Governance AI", page_icon="🧭", layout="wide")

    def _err(msg: str):
        try:
            st.toast(msg, icon="⚠️")
        except Exception:
            st.warning(msg)

    st.title("Support RAG per YouTrack + Governance Monitor")
    st.caption("Gestione ticket di supporto basata su storico YouTrack + RAG + LLM; monitor AI per governance.")

    with st.sidebar:
        st.header("Connessione YouTrack")
        yt_url = st.text_input("Server URL", placeholder="https://<org>.myjetbrains.com/youtrack", value=os.getenv("YT_BASE_URL", ""))
        yt_token = st.text_input("Token (Bearer)", type="password", value=os.getenv("YT_TOKEN", ""))
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
        st.header("Governance: Azioni")
        do_comment = st.checkbox("Pubblica commento AI", value=False)
        do_write_autoflag = st.checkbox("Scrivi AutoFlag (computed)", value=False, help="Aggiorna AutoFlag con Overdue/DelayRisk/None")

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

    # Tabs per separare RAG e Governance
    tab_rag, tab_gov = st.tabs(["RAG Support", "Governance Monitor (AI)"])

    with tab_rag:
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
                st.dataframe(df, width=st.session_state.get("page_width", 1200))
            else:
                st.caption("Nessun ticket caricato")

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

                    st.text("Risposta")
                    st.write(answer)

                    with st.expander("Ticket simili usati come contesto"):
                        for (doc, meta, dist) in retrieved:
                            st.write(f"[{meta.get('id_readable','')}] {meta.get('summary','')} | distanza={dist:.3f}")
                            st.code(doc[:1200])

                except Exception as e:
                    _err(f"Errore chatbot: {e}")

    with tab_gov:
        st.subheader("Governance Monitor (AI)")
        st.caption("Analizza scadenze e bloccanti del progetto di governance. Non tocca i ticket a meno che tu lo chieda.")

        proj_key_monitor = st.text_input("Project key da monitorare", value="ROUT")
        run_monitor = st.button("Analizza rischio")

        yt_client = st.session_state.get("yt_client")
        if not yt_client:
            st.info("Connetti YouTrack dalla sidebar per abilitare il monitor.")
        elif run_monitor:
            try:
                gov_issues = yt_client.list_issues_governance(proj_key_monitor, limit=1000)

                provider_key = "openai" if llm_provider.lower().startswith("openai") else "ollama"
                llm = LLMBackend(provider=provider_key, model=llm_model, temperature=temperature)

                rows = []
                for it in gov_issues:
                    flag = compute_flag(it)
                    suggested = ai_assess_actions(llm, it) if flag in ("Overdue", "DelayRisk") or (it.get("blocked") == "Yes") else ""

                    if do_write_autoflag:
                        try:
                            yt_client.set_autoflag(it["id"], flag if flag != "None" else None)
                        except Exception as e:
                            st.toast(f"AutoFlag fallito {it['id_readable']}: {e}", icon="⚠️")

                    if do_comment and suggested:
                        try:
                            yt_client.add_comment(it["id"], f"[AI] {suggested}")
                        except Exception as e:
                            st.toast(f"Commento fallito {it['id_readable']}: {e}", icon="⚠️")

                    rows.append({
                        "ID": it.get("id_readable"),
                        "Summary": it.get("summary"),
                        "Type": it.get("type"),
                        "Team": it.get("owner_team"),
                        "Due": _date_to_str(_ms_to_date(it.get("due_ms"))),
                        "SLA": _date_to_str(_ms_to_date(it.get("sla_ms"))),
                        "Blocked": it.get("blocked"),
                        "AutoFlag": it.get("autoflag"),
                        "ComputedFlag": flag,
                        "AI Actions": suggested,
                    })

                st.success(f"Analizzati {len(rows)} ticket nel progetto {proj_key_monitor}")
                df = pd.DataFrame(rows)

                col1, col2, col3, col4 = st.columns(4)
                overdue_n = (df["ComputedFlag"] == "Overdue").sum()
                delay_n = (df["ComputedFlag"] == "DelayRisk").sum()
                blocked_n = (df["Blocked"] == "Yes").sum()
                col1.metric("Overdue", int(overdue_n))
                col2.metric("DelayRisk (≤3g)", int(delay_n))
                col3.metric("Blocked", int(blocked_n))
                col4.metric("Totale", int(len(df)))

                st.dataframe(df, use_container_width=True)
            except Exception as e:
                _err(f"Errore monitor: {e}")

    st.markdown("---")
    st.caption("Suggerimenti: usa embedding locali per privacy; job periodico per sincronizzare i ticket; rating risposte per migliorare i prompt; usa AutoFlag con parsimonia.")

# ------------------------------
# CLI help + Self tests (senza Streamlit)
# ------------------------------

def _cli_help() -> None:
    print("\n=== Support RAG + Governance Monitor (CLI) ===")
    print("Streamlit non è installato nell'ambiente corrente.")
    print("Installa i requisiti e avvia l'app web:")
    print("  pip install -U streamlit requests chromadb sentence-transformers openai tiktoken pandas")
    print("  # opzionale per modelli locali: pip install -U ollama")
    print("  streamlit run app_gov.py\n")


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
