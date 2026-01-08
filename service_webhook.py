# service_webhook.py
# Comments in English, as requested.
#
# Multi-project webhook:
# - Resolves project config from an on-disk registry (no environment variables required)
# - Uses the same Chroma + retrieval logic as the Streamlit UI (rag_core.py)
# - Posts "similar tickets" suggestions as a YouTrack comment

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from rag_core import load_chroma_collection, RetrievalConfig, retrieve_and_build_prompt

# ----------------------------
# Paths (no env vars required)
# ----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(APP_DIR, ".taai_projects.json")
DEFAULT_PROJECTS_DIR = os.path.join(APP_DIR, "projects")

# Comment marker used for idempotency (avoid duplicates)
RAG_MARKER = "<!--RAG_SIMILAR_v1-->"

# ----------------------------
# Models
# ----------------------------
class IssueCreatedPayload(BaseModel):
    issue_id: str
    id_readable: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None

    # Optional explicit routing (recommended if issue_id is an internal id)
    project_key: Optional[str] = None
    project_short_name: Optional[str] = None

    # Optional override for KB selection
    kb_collection: Optional[str] = None


@dataclass
class ProjectConfig:
    project_key: str
    yt_url: str
    yt_token: str
    webhook_secret: str
    persist_dir: str

    # Optional secrets for embeddings (OpenAI)
    openai_api_key: str = ""

    # Retrieval knobs
    top_k: int = 5
    max_distance: float = 0.9
    collapse_duplicates: bool = True
    per_parent_display: int = 1
    per_parent_prompt: int = 3
    stitch_max_chars: int = 1600

    # Embeddings defaults (fallback when collection meta is missing)
    emb_backend: str = "OpenAI"
    emb_model_name: str = "text-embedding-3-small"

    # Memory (optional)
    enable_memory: bool = False
    mem_collection: str = "memories"
    mem_ttl_days: int = 180
    mem_cap: int = 2


# ----------------------------
# Embeddings
# ----------------------------
class Embedder:
    """Simple embedding wrapper supporting OpenAI or sentence-transformers."""

    def __init__(self, provider: str, model: str, *, openai_api_key: str = ""):
        self.provider = (provider or "").strip().lower()
        self.model = (model or "").strip()
        self.openai_api_key = (openai_api_key or "").strip()

        if self.provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError("OpenAI embeddings selected but 'openai_api_key' is missing in the project config.")
            from openai import OpenAI  # lazy import
            self.client = OpenAI(api_key=self.openai_api_key)
        elif self.provider in ("sentence-transformers", "sentence_transformers", "local", "st"):
            from sentence_transformers import SentenceTransformer  # lazy import
            self.model_st = SentenceTransformer(self.model or "all-MiniLM-L6-v2")
        else:
            raise RuntimeError(f"Unsupported embeddings provider: {self.provider}")

    def embed_one(self, text: str) -> List[float]:
        text = text or ""
        if self.provider == "openai":
            res = self.client.embeddings.create(model=self.model, input=[text])
            return res.data[0].embedding  # type: ignore
        vec = self.model_st.encode([text], normalize_embeddings=True)[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)


# ----------------------------
# Registry + config loading
# ----------------------------
def _normalize_project_key(k: str) -> str:
    return (k or "").strip().lower().replace(" ", "_")


def _load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        return {}
    return {}


def _load_registry() -> Dict[str, Any]:
    reg = _load_json(REGISTRY_PATH)
    if reg.get("projects") is None:
        reg["projects"] = {}
    return reg


def _get_project_prefs_path(registry: Dict[str, Any], project_key: str) -> str:
    project_key = _normalize_project_key(project_key)
    pinfo = (registry.get("projects") or {}).get(project_key) or {}
    prefs_path = (pinfo.get("prefs_path") or "").strip()
    if prefs_path:
        if not os.path.isabs(prefs_path):
            prefs_path = os.path.join(APP_DIR, prefs_path)
        return prefs_path
    return os.path.join(DEFAULT_PROJECTS_DIR, project_key, "app_prefs.json")


def _parse_project_short_name_from_readable(s: str) -> Optional[str]:
    s = (s or "").strip()
    if "-" not in s:
        return None
    left = s.split("-", 1)[0].strip()
    return left or None


def _resolve_project_key_from_payload(payload: IssueCreatedPayload) -> Optional[str]:
    # Explicit routing always wins
    for k in [payload.project_key, payload.project_short_name]:
        if k and k.strip():
            return _normalize_project_key(k)

    # Try parsing NETKB-3 -> netkb
    for s in [payload.id_readable, payload.issue_id]:
        short = _parse_project_short_name_from_readable(s or "")
        if short:
            return _normalize_project_key(short)

    return None


def _load_project_config(project_key: str) -> ProjectConfig:
    registry = _load_registry()
    prefs_path = _get_project_prefs_path(registry, project_key)
    prefs = _load_json(prefs_path)

    yt_url = (prefs.get("yt_url") or "").rstrip("/")
    yt_token = (prefs.get("yt_token") or "").strip()
    webhook_secret = (prefs.get("webhook_secret") or "").strip()

    persist_dir = (prefs.get("persist_dir") or "").strip()
    # If persist_dir is relative (project-scoped), resolve it relative to the project prefs folder,
    # not relative to APP_DIR. This allows project-scoped folders like:
    #   projects/<project_key>/chroma
    prefs_dir = os.path.dirname(prefs_path)
    if persist_dir and not os.path.isabs(persist_dir):
        persist_dir = os.path.join(prefs_dir, persist_dir)

    if not persist_dir:
        # Default to a project-scoped folder
        persist_dir = os.path.join(os.path.dirname(prefs_path), "chroma")

    if not yt_url:
        raise RuntimeError(f"Project '{project_key}': missing 'yt_url' in {prefs_path}")
    if not yt_token:
        raise RuntimeError(f"Project '{project_key}': missing 'yt_token' in {prefs_path}")
    if not webhook_secret:
        raise RuntimeError(f"Project '{project_key}': missing 'webhook_secret' in {prefs_path}")

    return ProjectConfig(
        project_key=_normalize_project_key(project_key),
        yt_url=yt_url,
        yt_token=yt_token,
        webhook_secret=webhook_secret,
        persist_dir=persist_dir,
        openai_api_key=(prefs.get("openai_api_key") or "").strip(),
        top_k=int(prefs.get("top_k", 5)),
        max_distance=float(prefs.get("max_distance", 0.9)),
        collapse_duplicates=bool(prefs.get("collapse_duplicates", True)),
        per_parent_display=int(prefs.get("per_parent_display", 1)),
        per_parent_prompt=int(prefs.get("per_parent_prompt", 3)),
        stitch_max_chars=int(prefs.get("stitch_max_chars", 1600)),
        emb_backend=str(prefs.get("emb_backend", "OpenAI")),
        emb_model_name=str(prefs.get("emb_model_name", "text-embedding-3-small")),
        enable_memory=bool(prefs.get("enable_memory", False)),
        mem_collection=str(prefs.get("mem_collection", "memories")),
        mem_ttl_days=int(prefs.get("mem_ttl_days", 180)),
        mem_cap=int(prefs.get("mem_cap", 2)),
    )


def _read_collection_meta(persist_dir: str, collection: str) -> Tuple[Optional[str], Optional[str]]:
    """Read provider/model used for indexing (same logic as app.py stores)."""
    meta_path = os.path.join(persist_dir, f"{collection}__meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            provider = (m.get("provider") or "").strip().lower()
            model = (m.get("model") or "").strip()
            if provider and model:
                return provider, model
        except Exception:
            pass
    return None, None


def _backend_to_provider(emb_backend: str) -> str:
    b = (emb_backend or "").strip().lower()
    if b in ("openai",):
        return "openai"
    if b in ("sentence-transformers", "sentence_transformers", "local", "st"):
        return "sentence-transformers"
    return "openai"


# ----------------------------
# YouTrack helpers (stateless)
# ----------------------------
def _yt_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def yt_get_issue(yt_url: str, token: str, issue_id: str) -> Dict[str, Any]:
    fields = "id,idReadable,summary,description,project(shortName,name)"
    url = f"{yt_url}/api/issues/{issue_id}"
    r = requests.get(url, headers=_yt_headers(token), params={"fields": fields}, timeout=30)
    r.raise_for_status()
    return r.json()


def yt_list_comments(yt_url: str, token: str, issue_id: str) -> List[Dict[str, Any]]:
    url = f"{yt_url}/api/issues/{issue_id}/comments"
    r = requests.get(url, headers=_yt_headers(token), params={"fields": "id,text"}, timeout=30)
    r.raise_for_status()
    return r.json() or []


def yt_add_comment(yt_url: str, token: str, issue_id: str, text: str) -> None:
    url = f"{yt_url}/api/issues/{issue_id}/comments"
    r = requests.post(url, headers=_yt_headers(token), json={"text": text}, timeout=30)
    r.raise_for_status()


# ----------------------------
# Comment formatting
# ----------------------------
def _format_similar_comment(items: List[Tuple[str, Dict[str, Any], float, str]], cfg: ProjectConfig) -> str:
    """Format a YouTrack comment listing the most similar tickets."""
    if not items:
        return "No similar tickets found.\n\n" + RAG_MARKER

    lines: List[str] = []
    lines.append("Similar tickets found in the Knowledge Base:")
    lines.append("")

    for i, (_doc, meta, dist, _src) in enumerate(items, start=1):
        issue = (meta or {}).get("id_readable") or (meta or {}).get("key") or "UNKNOWN"
        summary = (meta or {}).get("summary") or ""
        summary = summary.replace("\n", " ").strip()
        if len(summary) > 140:
            summary = summary[:137] + "..."
        link = f"{cfg.yt_url}/issue/{issue}"
        lines.append(f"{i}) {issue} — {summary}")
        lines.append(f"   {link}")
        lines.append(f"   distance: {dist:.3f}")
        lines.append("")

    lines.append(RAG_MARKER)
    return "\n".join(lines)


def _has_marker_comment(comments: List[Dict[str, Any]]) -> bool:
    return any(RAG_MARKER in (c.get("text") or "") for c in (comments or []))


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="TicketsAnalyzerAI Webhook", version="2.0")


@app.post("/yt/issue-created")
def issue_created(payload: IssueCreatedPayload, x_webhook_secret: Optional[str] = Header(default=None)):
    issue_id = (payload.issue_id or "").strip()
    if not issue_id:
        raise HTTPException(status_code=400, detail="issue_id is required")

    # Resolve project config
    project_key = _resolve_project_key_from_payload(payload)
    if not project_key:
        registry = _load_registry()
        projects = sorted((registry.get("projects") or {}).keys())
        if len(projects) == 1:
            project_key = projects[0]
        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot resolve project. Provide payload.project_key/project_short_name or id_readable like 'NETKB-3'.",
            )

    try:
        cfg = _load_project_config(project_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project config for '{project_key}': {e}")

    # Project-scoped webhook secret validation
    if cfg.webhook_secret:
        if not x_webhook_secret or x_webhook_secret != cfg.webhook_secret:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    # Fetch issue if needed
    id_readable = (payload.id_readable or "").strip()
    summary = (payload.summary or "").strip()
    description = (payload.description or "").strip()

    if not (summary or description):
        try:
            issue_obj = yt_get_issue(cfg.yt_url, cfg.yt_token, issue_id)
            id_readable = id_readable or (issue_obj.get("idReadable") or "")
            summary = summary or (issue_obj.get("summary") or "")
            description = description or (issue_obj.get("description") or "")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch issue from YouTrack: {e}")

    query_text = (summary + "\n\n" + description).strip()

    # Resolve KB collection:
    # - payload override wins
    # - otherwise use issue prefix/project key (NETKB-3 -> 'netkb')
    if kb_collection and kb_collection.strip():
        kb_name = kb_collection.strip().lower()
    else:
        short = _parse_project_short_name_from_readable(id_readable) or _parse_project_short_name_from_readable(issue_id) or cfg.project_key
        kb_name = (short or cfg.project_key).strip().lower()

    # Open KB collection
    kb_coll = load_chroma_collection(cfg.persist_dir, kb_name, space="cosine")

    # Optional MEM collection
    mem_coll = None
    if cfg.enable_memory:
        try:
            mem_coll = load_chroma_collection(cfg.persist_dir, cfg.mem_collection, space="cosine")
        except Exception:
            mem_coll = None

    # Embeddings provider/model must match the collection used for indexing (meta file if present).
    provider, model = _read_collection_meta(cfg.persist_dir, kb_name)
    provider = provider or _backend_to_provider(cfg.emb_backend)
    model = model or cfg.emb_model_name

    # Build query embedding
    try:
        embedder = Embedder(provider, model, openai_api_key=cfg.openai_api_key)
        query_emb = embedder.embed_one(query_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {e}")

    # Build retrieval config
    rconf = RetrievalConfig(
        top_k=cfg.top_k,
        max_distance=cfg.max_distance,
        collapse_duplicates=cfg.collapse_duplicates,
        per_parent_display=cfg.per_parent_display,
        per_parent_prompt=cfg.per_parent_prompt,
        stitch_max_chars=cfg.stitch_max_chars,
    )

    now_ts = int(time.time())

    # Run retrieval
    prompt, merged_view, _merged_prompt = retrieve_and_build_prompt(
        query=query_text,
        query_embedding=query_emb,
        kb_collection=kb_coll,
        mem_collection=mem_coll,
        cfg=rconf,
        now_ts=now_ts,
        mem_cap=cfg.mem_cap,
    )

    # Remove self-hit (if metadata stores id_readable)
    if id_readable:
        merged_view = [
            (doc, meta, dist, src)
            for (doc, meta, dist, src) in merged_view
            if (meta or {}).get("id_readable") != id_readable
        ]

    # Idempotency: skip if marker comment already exists
    try:
        comments = yt_list_comments(cfg.yt_url, cfg.yt_token, issue_id)
        if _has_marker_comment(comments):
            return {
                "status": "skipped",
                "reason": "rag_comment_already_present",
                "issue_id": issue_id,
                "id_readable": id_readable,
                "project_key": cfg.project_key,
                "kb_collection": kb_name,
                "found": len(merged_view),
            }
    except Exception:
        pass  # fail-open

    # Format + post comment
    comment_text = _format_similar_comment(merged_view, cfg)
    try:
        yt_add_comment(cfg.yt_url, cfg.yt_token, issue_id, comment_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to add comment to YouTrack: {e}")

    return {
        "status": "ok",
        "issue_id": issue_id,
        "id_readable": id_readable,
        "project_key": cfg.project_key,
        "kb_collection": kb_name,
        "persist_dir": cfg.persist_dir,
        "provider": provider,
        "model": model,
        "found": len(merged_view),
        "prompt_preview": prompt[:400],
    }
