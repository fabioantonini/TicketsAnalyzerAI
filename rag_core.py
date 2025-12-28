"""rag_core.py

Shared, UI-agnostic core utilities for:
- Vector DB (Chroma) initialization
- Retrieval post-processing (collapse, stitching)
- Prompt building

Keep this module free from Streamlit specifics so it can be reused by:
- Streamlit UI (app.py)
- FastAPI webhook service (YouTrack trigger)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

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


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval and post-processing."""
    top_k: int = 5
    max_distance: float = 0.35
    collapse_duplicates: bool = True
    per_parent_display: int = 1
    per_parent_prompt: int = 3
    stitch_max_chars: int = 1500


def get_chroma_client(persist_dir: str):
    """Return a Chroma PersistentClient (creating the folder if needed).

    NOTE: This function intentionally avoids importing Streamlit.
    Caller should handle UI-level error reporting.
    """
    import chromadb  # lazy import
    from chromadb.config import Settings as ChromaSettings

    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def load_chroma_collection(persist_dir: str, collection_name: str, space: str = "cosine"):
    """Open (or create) and return the Chroma collection.

    Note: the Streamlit app expects a *collection* object (with .add/.query).
    If you need the client as well, call get_chroma_client() separately.
    """
    client = get_chroma_client(persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": space},
    )


def load_chroma_client_and_collection(
    persist_dir: str, collection_name: str, space: str = "cosine"
):
    """Convenience helper returning (client, collection)."""
    client = get_chroma_client(persist_dir)
    coll = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": space},
    )
    return client, coll


def build_prompt(user_ticket: str, retrieved: List[Tuple[str, dict, float]]) -> str:
    """Create the LLM prompt given the new ticket and retrieved contexts."""
    parts = [
        "New ticket:\n" + (user_ticket or "").strip(),
        "\nSimilar tickets found (closest first):",
    ]
    for doc, meta, dist in retrieved:
        meta = meta or {}
        parts.append(
            f"- {meta.get('id_readable','')} | distance={dist:.3f} | {meta.get('summary','')}\n{(doc or '')[:500]}"
        )
    parts.append(
        "\nAnswer including citations in [ ] of the relevant ticket IDs. Please use always english in the answer"
    )
    return "\n".join(parts)


def collapse_by_parent(
    results: List[Tuple[str, dict, float, str]],
    per_parent: int = 1,
    stitch_for_prompt: bool = False,
    max_chars: int = 1200,
) -> List[Tuple[str, dict, float, str]]:
    """Collapse multiple chunks from the same ticket into one row.

    results: list of (doc, meta, dist, src)
    - per_parent: keep at most N items per parent (1 for display)
    - stitch_for_prompt: if True, concatenate selected chunks in order of 'pos' (bounded by max_chars)
    Returns: list of (doc, meta, dist, src) sorted by distance.
    """
    groups: Dict[str, List[Tuple[str, dict, float, str]]] = defaultdict(list)
    for doc, meta, dist, src in results:
        meta = meta or {}
        pid = meta.get("parent_id") or meta.get("id_readable") or "unknown"
        groups[str(pid)].append((doc, meta, dist, src))

    collapsed: List[Tuple[str, dict, float, str]] = []
    for _pid, items in groups.items():
        items = sorted(items, key=lambda x: (x[2], x[1].get("pos", 0)))  # dist, then pos
        keep = items[: max(1, int(per_parent))]

        if stitch_for_prompt and len(keep) > 1:
            keep_sorted = sorted(keep, key=lambda x: x[1].get("pos", 0))
            buf: List[str] = []
            total = 0
            for d, _m, _dist, _src in keep_sorted:
                d = d or ""
                if total + len(d) + 2 > int(max_chars):
                    break
                buf.append(d)
                total += len(d) + 2
            best = keep[0]
            stitched = "\n\n".join(buf) if buf else (best[0] or "")
            collapsed.append((stitched, best[1], best[2], best[3]))
        else:
            collapsed.append(keep[0])

    return sorted(collapsed, key=lambda x: x[2])


def retrieve_and_build_prompt(
    *,
    query: str,
    query_embedding: List[float],
    kb_collection,
    mem_collection=None,
    cfg: RetrievalConfig = RetrievalConfig(),
    now_ts: Optional[int] = None,
    mem_cap: int = 2,
) -> Tuple[str, List[Tuple[str, dict, float, str]], List[Tuple[str, dict, float, str]]]:
    """Run retrieval on KB (+ optional memory collection) and build the LLM prompt.

    Returns:
    - prompt (str)
    - merged_collapsed_view (for UI display)
    - merged_for_prompt (actually used as context)
    """
    q_emb = [query_embedding]

    kb_res = kb_collection.query(query_embeddings=q_emb, n_results=int(cfg.top_k))
    kb_docs = kb_res.get("documents", [[]])[0]
    kb_metas = kb_res.get("metadatas", [[]])[0]
    kb_dists = kb_res.get("distances", [[]])[0]

    kb_retrieved = [
        (doc, meta, dist, "KB")
        for doc, meta, dist in zip(kb_docs, kb_metas, kb_dists)
        if dist is not None and float(dist) <= float(cfg.max_distance)
    ]

    mem_retrieved: List[Tuple[str, dict, float, str]] = []
    if mem_collection is not None:
        mem_res = mem_collection.query(query_embeddings=q_emb, n_results=min(5, int(cfg.top_k)))
        mem_docs = mem_res.get("documents", [[]])[0]
        mem_metas = mem_res.get("metadatas", [[]])[0]
        mem_dists = mem_res.get("distances", [[]])[0]

        for doc, meta, dist in zip(mem_docs, mem_metas, mem_dists):
            if dist is None:
                continue
            meta = meta or {}
            # TTL filter is optional
            if now_ts is not None:
                exp = int(meta.get("expires_at", 0) or 0)
                if exp and exp < int(now_ts):
                    continue
            if float(dist) <= float(cfg.max_distance):
                mem_retrieved.append((doc, meta, float(dist), "MEM"))

    merged = sorted(mem_retrieved, key=lambda x: x[2])[: int(mem_cap)] + sorted(
        kb_retrieved, key=lambda x: x[2]
    )[: max(0, int(cfg.top_k) - min(int(mem_cap), len(mem_retrieved)))]

    if merged:
        merged_collapsed_view = collapse_by_parent(
            merged,
            per_parent=int(cfg.per_parent_display),
            stitch_for_prompt=False,
        )
        merged_for_prompt = collapse_by_parent(
            merged,
            per_parent=int(cfg.per_parent_prompt),
            stitch_for_prompt=True,
            max_chars=int(cfg.stitch_max_chars),
        )
        retrieved_for_prompt = [(doc, meta, dist) for (doc, meta, dist, _src) in merged_for_prompt]
        prompt = build_prompt(query, retrieved_for_prompt)
    else:
        merged_collapsed_view = []
        merged_for_prompt = []
        prompt = (
            f"New ticket:\n{(query or '').strip()}\n\n"
            "No similar ticket was found in the knowledge base."
        )

    return prompt, merged_collapsed_view, merged_for_prompt
