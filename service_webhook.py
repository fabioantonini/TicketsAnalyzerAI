# service_webhook.py
# Comments in English, as requested.

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from rag_core import (
    load_chroma_collection,
    RetrievalConfig,
    retrieve_and_build_prompt,
)

# ----------------------------
# Configuration (env vars)
# ----------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", os.getenv("PERSIST_DIR", "./data/chroma"))
KB_COLLECTION = os.getenv("KB_COLLECTION", "tickets")
MEM_COLLECTION = os.getenv("MEM_COLLECTION", "memories")  # optional
ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "0").strip() == "1"

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

YT_BASE_URL = os.getenv("YT_BASE_URL", "").rstrip("/")
YT_TOKEN = os.getenv("YT_TOKEN", "").strip()

# Retrieval knobs
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.9"))
PER_PARENT_DISPLAY = int(os.getenv("PER_PARENT_DISPLAY", "1"))
PER_PARENT_PROMPT = int(os.getenv("PER_PARENT_PROMPT", "3"))
STITCH_MAX_CHARS = int(os.getenv("STITCH_MAX_CHARS", "1500"))
MEM_CAP = int(os.getenv("MEM_CAP", "2"))

# OpenAI embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY_EXPERIMENTS", "")).strip()

# ----------------------------
# Minimal embedder (same behavior as UI)
# ----------------------------
class Embedder:
    """Simple embedding wrapper supporting OpenAI or sentence-transformers."""

    def __init__(self, provider: str, model: str):
        self.provider = (provider or "").strip().lower()
        self.model = (model or "").strip()

        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set")
            from openai import OpenAI  # lazy import
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider in ("sentence-transformers", "local", "st"):
            from sentence_transformers import SentenceTransformer  # lazy import
            self.model_st = SentenceTransformer(self.model or "all-MiniLM-L6-v2")
        else:
            raise RuntimeError(f"Unsupported embeddings provider: {self.provider}")

    def embed_one(self, text: str) -> List[float]:
        text = text or ""
        if self.provider == "openai":
            res = self.client.embeddings.create(model=self.model, input=[text])
            return res.data[0].embedding  # type: ignore
        # sentence-transformers
        vec = self.model_st.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()


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


def _yt_headers() -> Dict[str, str]:
    if not YT_BASE_URL or not YT_TOKEN:
        raise RuntimeError("YT_BASE_URL and/or YT_TOKEN not configured")
    return {"Authorization": f"Bearer {YT_TOKEN}", "Content-Type": "application/json"}


def yt_get_issue(issue_id: str) -> Dict[str, Any]:
    """Fetch issue details from YouTrack by internal issue id."""
    url = f"{YT_BASE_URL}/api/issues/{issue_id}"
    params = {"fields": "id,idReadable,summary,description,project(shortName,name)"}
    r = requests.get(url, headers=_yt_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def yt_has_rag_comment(issue_id: str) -> bool:
    url = f"{YT_BASE_URL}/api/issues/{issue_id}/comments"
    params = {"fields": "text"}
    r = requests.get(url, headers=_yt_headers(), params=params, timeout=30)
    r.raise_for_status()
    for c in r.json():
        if "<!--RAG_SIMILAR_v1-->" in (c.get("text") or ""):
            return True
    return False

def yt_add_comment(issue_id: str, text: str) -> None:
    """Add comment to issue."""
    url = f"{YT_BASE_URL}/api/issues/{issue_id}/comments"
    payload = {"text": text}
    r = requests.post(url, headers=_yt_headers(), json=payload, timeout=30)
    r.raise_for_status()


def _format_similar_comment(
    issue_id_readable: str,
    merged_collapsed_view: List[Tuple[str, dict, float, str]],
    yt_base_url: str,
) -> str:
    """Build a readable comment. Keep it concise."""
    lines = []
    lines.append(f"🤖 Similar tickets suggestion for **{issue_id_readable}**")
    lines.append("")
    if not merged_collapsed_view:
        lines.append("No similar tickets found in the knowledge base (within the configured threshold).")
        return "\n".join(lines)

    lines.append("Top similar items:")
    for doc, meta, dist, src in merged_collapsed_view:
        if src != "KB":
            # Optionally include MEM items; usually keep KB only in comments
            continue
        meta = meta or {}
        rid = (meta.get("id_readable") or "").strip()
        summ = (meta.get("summary") or "").strip()
        if rid:
            link = f"{yt_base_url.rstrip('/')}/issue/{rid}" if yt_base_url else ""
            label = f"{rid} — {summ}" if summ else rid
            if link:
                lines.append(f"- {label} (distance={dist:.3f}) → {link}")
            else:
                lines.append(f"- {label} (distance={dist:.3f})")
        else:
            # fallback
            preview = (doc or "").strip().replace("\n", " ")
            lines.append(f"- (distance={dist:.3f}) {preview[:140]}...")

    lines.append("")
    lines.append("_Generated automatically on issue creation._")
    lines.append("")
    lines.append("<!--RAG_SIMILAR_v1-->")
    return "\n".join(lines)


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="YouTrack RAG Webhook", version="1.0")


class IssueCreatedPayload(BaseModel):
    # Prefer internal issue id from YouTrack workflow
    issue_id: str

    # Optional fields (you can send them directly from workflow to avoid GET)
    id_readable: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None

    # Optional override
    kb_collection: Optional[str] = None


@app.post("/yt/issue-created")
def issue_created(
    payload: IssueCreatedPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    # 1) Security check (optional but recommended)
    if WEBHOOK_SECRET:
        if not x_webhook_secret or x_webhook_secret != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    # 2) Get issue text
    issue_id = payload.issue_id.strip()
    if not issue_id:
        raise HTTPException(status_code=400, detail="issue_id is required")

    id_readable = (payload.id_readable or "").strip()
    summary = (payload.summary or "").strip()
    description = (payload.description or "").strip()

    if not (summary or description):
        # Fetch from YouTrack if not provided
        try:
            issue = yt_get_issue(issue_id)
            id_readable = id_readable or (issue.get("idReadable") or "")
            summary = summary or (issue.get("summary") or "")
            description = description or (issue.get("description") or "")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch issue from YouTrack: {e}")

    query_text = (summary + "\n\n" + description).strip()

    # 3) Load KB (and optional MEM) collections via rag_core
    kb_name = (payload.kb_collection or KB_COLLECTION).strip() or KB_COLLECTION
    kb_coll = load_chroma_collection(CHROMA_DIR, kb_name, space="cosine")

    mem_coll = None
    if ENABLE_MEMORY:
        try:
            mem_coll = load_chroma_collection(CHROMA_DIR, MEM_COLLECTION, space="cosine")
        except Exception:
            mem_coll = None  # do not fail the webhook if MEM is not available

    # 4) Ensure embeddings match the collection used for indexing
    provider, model = _read_collection_meta(CHROMA_DIR, kb_name)
    provider = provider or "openai"
    model = model or "text-embedding-3-small"

    try:
        embedder = Embedder(provider=provider, model=model)
        q_vec = embedder.embed_one(query_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 5) Retrieval + prompt (prompt is available if you later want to auto-answer too)
    cfg = RetrievalConfig(
        top_k=TOP_K,
        max_distance=MAX_DISTANCE,
        collapse_duplicates=True,
        per_parent_display=PER_PARENT_DISPLAY,
        per_parent_prompt=PER_PARENT_PROMPT,
        stitch_max_chars=STITCH_MAX_CHARS,
    )

    prompt, merged_view, _merged_prompt = retrieve_and_build_prompt(
        query=query_text,
        query_embedding=q_vec,
        kb_collection=kb_coll,
        mem_collection=mem_coll,
        cfg=cfg,
        now_ts=int(time.time()),
        mem_cap=MEM_CAP,
    )

    # Remove self-reference (same issue)
    self_ids = {issue_id, id_readable}
    merged_view = [
        (doc, meta, dist, src)
        for (doc, meta, dist, src) in merged_view
        if (meta or {}).get("id_readable") not in self_ids
    ]

    # 6) Write-back: comment into YouTrack
    try:
        comment_text = _format_similar_comment(
            issue_id_readable=id_readable or issue_id,
            merged_collapsed_view=merged_view,
            yt_base_url=YT_BASE_URL,
        )
        if yt_has_rag_comment(issue_id):
            return {
                "status": "skipped",
                "reason": "rag_comment_already_present",
                "issue_id": issue_id,
            }
        yt_add_comment(issue_id, comment_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to add comment in YouTrack: {e}")

    return {
        "status": "ok",
        "issue_id": issue_id,
        "id_readable": id_readable,
        "kb_collection": kb_name,
        "provider": provider,
        "model": model,
        "found": len(merged_view),
        # helpful for debugging / future extensions:
        "prompt_preview": prompt[:400],
    }
