"""rag_core.py

Shared RAG logic extracted from app.py.

Design constraints
- No Streamlit imports.
- Keep behavior compatible with the previous inline implementation in app.py.
- Make it easy to reuse from a webhook service (FastAPI) later.

All code comments are in English.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


RetrievedRow = Tuple[str, dict, float, str]


def build_prompt(user_ticket: str, retrieved: List[Tuple[str, dict, float]]) -> str:
    """Build the exact same prompt shape used in app.py."""
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
    results: Sequence[RetrievedRow],
    per_parent: int = 1,
    stitch_for_prompt: bool = False,
    max_chars: int = 1200,
) -> List[RetrievedRow]:
    """Collapse multiple chunks from the same ticket into one row.

    results: list of (doc, meta, dist, src)
    - per_parent: keep at most N items per parent (1 for display)
    - stitch_for_prompt: if True, concatenate selected chunks in order of 'pos' (bounded by max_chars)
    Returns: list of (doc, meta, dist, src) sorted by distance.
    """

    groups: Dict[str, List[RetrievedRow]] = defaultdict(list)
    for doc, meta, dist, src in results:
        meta = meta or {}
        pid = meta.get("parent_id") or meta.get("id_readable") or ""
        groups[str(pid)].append((doc, meta, dist, src))

    collapsed: List[RetrievedRow] = []
    for _pid, items in groups.items():
        items = sorted(items, key=lambda x: (x[2], x[1].get("pos", 0)))  # distance, then pos
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
            best = keep[0]  # best distance
            stitched = "\n\n".join(buf) if buf else (best[0] or "")
            collapsed.append((stitched, best[1], best[2], best[3]))
        else:
            collapsed.append(keep[0])

    return sorted(collapsed, key=lambda x: x[2])


@dataclass
class RetrievalConfig:
    top_k: int = 5
    max_distance: float = 0.35
    enable_memory: bool = False
    mem_collection: str = "__memories__"
    mem_cap: int = 2
    per_parent_display: int = 1
    per_parent_prompt: int = 3
    stitch_max_chars: int = 1500


def retrieve_and_build_prompt(
    *,
    query_text: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    get_chroma_client: Callable[[str], object],
    persist_dir: str,
    collection_name: str,
    now_ts: Optional[Callable[[], int]] = None,
    cfg: Optional[RetrievalConfig] = None,
) -> Tuple[List[RetrievedRow], List[RetrievedRow], List[RetrievedRow], str, Dict[str, object]]:
    """Retrieve similar docs (KB + optional memories) and build the LLM prompt.

    Returns:
      merged: raw merged list of rows (doc, meta, dist, src)
      merged_collapsed_view: collapsed list for UI display
      merged_for_prompt: collapsed/stitched list for prompt context
      prompt: final prompt string
      debug: small dict with useful counters
    """

    cfg = cfg or RetrievalConfig()
    q_emb = embed_fn([query_text])

    # --- KB retrieval ---
    client = get_chroma_client(persist_dir)
    kb_coll = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    kb_res = kb_coll.query(query_embeddings=q_emb, n_results=int(cfg.top_k))
    kb_docs = (kb_res.get("documents") or [[]])[0]
    kb_metas = (kb_res.get("metadatas") or [[]])[0]
    kb_dists = (kb_res.get("distances") or [[]])[0]

    kb_retrieved: List[RetrievedRow] = []
    for doc, meta, dist in zip(kb_docs, kb_metas, kb_dists):
        if dist is None:
            continue
        if float(dist) <= float(cfg.max_distance):
            kb_retrieved.append((doc, meta or {}, float(dist), "KB"))

    # --- Memories retrieval (optional) ---
    mem_retrieved: List[RetrievedRow] = []
    if bool(cfg.enable_memory):
        try:
            mem_client = get_chroma_client(persist_dir)
            mem_coll = mem_client.get_or_create_collection(
                name=cfg.mem_collection,
                metadata={"hnsw:space": "cosine"},
            )

            mem_res = mem_coll.query(query_embeddings=q_emb, n_results=min(5, int(cfg.top_k)))
            mem_docs = (mem_res.get("documents") or [[]])[0]
            mem_metas = (mem_res.get("metadatas") or [[]])[0]
            mem_dists = (mem_res.get("distances") or [[]])[0]

            now = now_ts() if now_ts else 0
            for doc, meta, dist in zip(mem_docs, mem_metas, mem_dists):
                if dist is None:
                    continue
                meta = meta or {}
                exp = int(meta.get("expires_at", 0) or 0)
                if now and exp and exp < now:
                    continue
                if float(dist) <= float(cfg.max_distance):
                    mem_retrieved.append((doc, meta, float(dist), "MEM"))
        except Exception:
            # Keep identical spirit: memory retrieval errors must not break the flow.
            mem_retrieved = []

    # Merge MEM + KB (same strategy used in app.py)
    mem_cap = int(cfg.mem_cap)
    merged = sorted(mem_retrieved, key=lambda x: x[2])[:mem_cap] + sorted(kb_retrieved, key=lambda x: x[2])[
        : max(0, int(cfg.top_k) - min(mem_cap, len(mem_retrieved)))
    ]

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
        prompt = build_prompt(query_text, retrieved_for_prompt)
    else:
        merged_collapsed_view = []
        merged_for_prompt = []
        prompt = (
            f"New ticket:\n{(query_text or '').strip()}\n\n" "No similar ticket was found in the knowledge base."
        )

    try:
        kb_count = kb_coll.count()  # type: ignore[attr-defined]
    except Exception:
        kb_count = "N/A"

    debug: Dict[str, object] = {
        "kb_raw_n": len(kb_docs),
        "kb_count": kb_count,
        "kb_used_n": len(kb_retrieved),
        "mem_used_n": len(mem_retrieved),
        "merged_n": len(merged),
        "view_n": len(merged_collapsed_view),
        "prompt_ctx_n": len(merged_for_prompt),
        "kb_dists_head": kb_dists[:5] if isinstance(kb_dists, list) else kb_dists,
    }

    return merged, merged_collapsed_view, merged_for_prompt, prompt, debug
