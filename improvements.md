# Improvements

## Must have (for a convincing PoC)

### 1) Incremental sync and upsert

* Download more than **500 issues** with pagination (`$skip`/`$top`) and only recent changes (`updatedSince`).
* Avoid duplicates using **upsert** (or `add` with `try/except` on existing IDs) and save a `last_sync.json` in the `persist_dir`.
* Save in the **collection metadata** the **embedding model name** and verify consistency during query (if model changes, warn or create a new collection).
* *Note*: currently, you index `summary + description` correctly via `text_blob`, so full texts are included in the Vector DB.

### 2) Chunking of long tickets

* Split very large descriptions into **chunks** with overlap (e.g. **512–800 tokens** with **80** overlap).
* Suggested metadata: `parent_id`, `chunk_id`, `pos`.
* Improves **recall** and reduces “noise” from encyclopedic descriptions.

### 3) Better retrieval: MMR and mini re-ranking

* After Chroma **top-k** query, apply **MMR (Maximal Marginal Relevance)** on the client side to diversify results.
* Optional: use a lightweight **CrossEncoder** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank the top-20.

### 4) UX: links, filter, and feedback

* Make ticket IDs **clickable** to YouTrack (column with URL like `BASE_URL/youtrack/issue/ID` or `BASE_URL/issue/ID` depending on instance).
* Add a **text filter** for loaded tickets.
* **Thumbs up/down** on the answer and save feedback to CSV to improve the prompt and assess **top-k hit rate**.

### 5) OpenAI robustness and secrets

* Read the key from **`st.secrets`** or `OPENAI_API_KEY`, maintaining compatibility with current `env`.
* `LLMBackend`: keep **fallback** *Responses API → Chat Completions*, but make text extraction more robust.
* **Never** save the YouTrack token in Vector DB metadata.

### 6) Cache and performance

* Use `st.cache_resource` for **Chroma**, **SentenceTransformer**, and **YouTrackClient** clients.
* Use `st.cache_data` for **`list_projects`** / **`list_issues`**.
* Significantly improves reload responsiveness.

---

## Nice to have

### 7) Hybrid search

* **BM25/keywords pre-filter** on candidates (e.g. using `rapidfuzz` or `rank-bm25`), then **semantic rerank**.
* Useful for technical texts with “strong” terms.

### 8) Multilingual and configurable models

* Default embedding better suited for Italian: `paraphrase-multilingual-MiniLM-L12-v2` (selectable from UI).
* Keep compatibility with current `all-MiniLM-L6-v2`.

### 9) Prompting and security

* **Prompt injection guard**: wrap the context between “Context Begin/End” tags and always request citations `[TICKET-ID]` (already provided by the system prompt).
* Truncate output and invite user clarification when context is weak.

### 10) Integrated quick evaluations

* **“Eval”** tab with 5–10 golden queries (real FAQs) showing **precision@k**, **MRR**, **nDCG**.
* A chart + table are enough to “sell” the PoC.

### 11) UI polishing

* Loader and progress bar during ingestion; **badge** in sidebar with document count.
* Wide dataframe with configured columns (you already have `layout="wide"`: good).

---

## Some suggestions

1. **More robust OpenAI secrets and keys**.
2. **YouTrack pagination** and **incremental sync**.
3. **Chunking** during ingestion.
4. **MMR** post-query (retrieve neighbor embeddings or recompute client-side; reorder with `mmr(...)`).
5. **Links** to issues in the table.
6. **Fast cache**:

   * Decorate `list_projects` / `list_issues` with `st.cache_data`.
   * Build Chroma/SentenceTransformer with `st.cache_resource`.
7. **UI quality of life**:

   * `text_input` to filter displayed tickets (summary/description) *before* ingestion.
   * Replace `os._exit(0)` with a note (close from console) or a shutdown endpoint; `os._exit` may terminate without cleanup.

---

## Notes on current code

* UI is already **wide** and table uses a **parametric width**: good.
* Ingestion indexes **summary** and **description** (via `text_blob`): good.
* `LLMBackend`: great fallback **Responses → Chat Completions**; just make text extraction more robust and centralize key management.

---

## Memory Roadmap

### Level A — Sticky prefs (✅ done)

* Non-sensitive preferences in `.app_prefs.json`, automatic model reset on provider change, validations.
  *Nothing to do, maybe small UX tweaks.*

### Level B — “Solution memory” (playbook) (🟡 almost complete)

**Already implemented**

* Separate **`memories`** collection (Chroma) with **TTL**.
* Playbook saving (“Mark as resolved”) with **3–6 sentence condensation**.
* Combined retrieval **KB ⊕ MEM** (same threshold), provenance **[MEM]**, **preview/expander**.
* **“Saved Playbooks”** view (table), **delete-all**, sticky toggle to show/hide.

**Missing/recommended**

1. **Advanced quality and tagging**

   * `quality` field from UI: `verified | draft` (currently hardcoded).
   * Additional tags in save (CSV textbox).
   * Filter by `project`/`tags` in MEM retrieval (limit to current project if present).
2. **Per-playbook management**

   * **Delete single** (trash icon in table).
   * **Edit text** (expander + `st.text_area`) and optional **re-embed**.
3. **Embedder compatibility**

   * Metadata: `mem_embed_provider`, `mem_embed_model`.
   * Warning if MEM ≠ current embedder; **re-embed on-demand**.
4. **UX/Performance listing**

   * **Pagination** (batch 200) + “Load more”.
   * **Local search** (doc/tags/project).
   * **Export** CSV/Markdown.
5. **Merging and ranking**

   * **MMR** on KB⊕MEM merge to avoid redundancy.
   * Configurable **cap** MEM (0–3) + optional dedicated threshold.
6. **Confirmation & policy**

   * Dialog “Confirm to store?” (NO-PII reminder).
   * More “procedural” condensation template.
7. **Metrics (debug)**

   * Caption with MEM distances, embedding/query time.
   * Usage counter (`uses`) in metadata.

### Level C — “Structured facts” (🔜 to implement)

**Goal**
Store reliable **key–value pairs (facts)** separate from playbooks and insert them into the prompt as **structured context**.

**Proposed design**

* **Storage**: SQLite `facts.db`, table:

  ```sql
  facts(user TEXT, project TEXT, key TEXT, value TEXT,
        source_ticket TEXT, created_at INT, expires_at INT,
        confidence REAL DEFAULT 1.0,
        UNIQUE(user, project, key))
  ```

- **UI**: in chat, pill “➕ Add fact” → modal (project, key, value, TTL).
  List “Active facts for X” with **Edit / Forget / Extend TTL**.
- **Prompting**: before the answer, serialize facts in **2–4 lines** (`project=NETKB; gateway_vendor=Acme; ...`) and ask the model to **not contradict them**.
- **Expiration & guardrails**: TTL, `expires_at` filter, **no PII/secret** (basic regex).
- **Search**: use facts to **boost retrieval** (e.g. `product=XYZ` → favor docs with matching tags).
- **Operations**: `upsert` per-project key, button “Forget this fact”.

### Pragmatic plan (2 mini-sprints)

**Sprint 1 – Level B refinements**
Delete/edit/re-embed playbooks; UI `quality`; search/pagination; CSV export; embedder metadata + mismatch warning; MEM cap + MMR.

**Sprint 2 – Level C MVP**
SQLite + CRUD facts; prompt serialization; TTL/expiration; basic validations; (opt.) retrieval boost from facts.

---

## PDF in KB

Adding PDFs can be **very useful** — provided it’s done carefully to avoid “polluting” responses. In short: **separate source**, **proper chunking**, **clear provenance**, **controlled merging** with tickets.

### When it makes sense

* Manuals, runbooks, external KBs, RFCs, vendor guides → **yes**.
* Noisy attachments (raw logs, generic reports) → **no** or **low quality**.

### How to integrate without confusion

1. **Separate collection**: `docs` distinct from `tickets`/`memories`. Query as **KB ⊕ MEM ⊕ DOCS** with **DOCS cap** (1–2) and **dedicated threshold**.
2. **Chunking & overlap**: ~**500–1000 tokens**, **overlap 100–150**, preserve `page_number` and headings.
3. **Rich metadata**: `source="pdf"`, `title`, `page`, `project`, `product`, `tags`, `quality`, `uri`.
4. **Clear provenance in UI**: label **[DOC]** and link “PDF p.12”; toggle “**Include PDF**” in sidebar.
5. **Ranking/merge**: DOCS threshold ~ ticket or stricter, **MMR** on merge, **DOCS cap** 1–2.
6. **Embeddings model**: `all-MiniLM-L6-v2` is fine; for long technical documents consider `text-embedding-3-large`.
7. **Quality & security**: avoid non-OCR-scanned files; no PII/secret; `quality` and (opt.) **TTL**.

**Mini-roadmap**

* **PDF-1 (ingest)**: upload/folder “PDF Documents” → extraction → chunking → embeddings → `docs` collection (metadata `title,page,uri,project,tags,quality`).
* **PDF-2 (query & UI)**: merge **KB⊕MEM⊕DOCS**, DOCS cap=1–2, MMR on; toggle “Include PDF”; render `[DOC]`.

---

## Evaluation metrics

### 1) Evaluation dataset (gold set)

* **Collection** (50–200 cases): for each ticket *Q* define **`answer_gold`** and **`doc_gold`**; include both easy and noisy cases.
* **Normalization**: clean texts for matching.
* **Split**: 80% *dev*, 20% *hold-out*.

> If using playbooks, add `mem_gold` for ~30% of cases.

### 2) Retrieval evaluation (without LLM)

* Run **embedding + nearest neighbors** pipeline on `tickets` (and `memories` if active).
* KB metrics: **Recall@k**, **Hit@k**, **MRR@k**, **nDCG@k**.
* MEM metrics: **Hit@k(mem)**, **MEM Coverage**.
* Threshold analysis: **Recall@k** curve varying **max distance** and **top-k**; choose **operating point**.
* Technical A/B: compare embedder/parameters on same gold.

### 3) Generation evaluation (with LLM)

* **EM/F1** for short answers; **ROUGE-L** for narrative ones.
* **Groundedness & Faithfulness**: % of supported sentences, no-contradiction (NLI/regex), **attribution rate**.
* Targets: **EM/F1 ≥ 0.6**, **ROUGE-L ≥ 0.35**, **Groundedness ≥ 0.85**, **Contradiction ≈ 0%**.
* **Human eval**: grid 1–5 (Correctness/Completeness/Clarity/Actionability), 2 evaluators over 30–50 cases.

### 4) KB ⊕ MEM merge evaluation

* Compare: (1) KB only, (2) KB⊕MEM cap 1, (3) KB⊕MEM cap 2 (+MMR).
* Measure: **F1/ROUGE**, **Groundedness**, **average time**; accept variant improving quality without too much latency (+10–20%).

### 5) Robustness & regressions

* Adversarial tests (vague/off-topic/typo queries) → clean “no match” outputs.
* Stability: repeat query (same temperature) and check variance.
* Benchmark script generates **CSV/HTML report** after each change.

### 6) Operational telemetry (prod-like)

* Logging (opt-in): query, k, threshold, docs used, distance, E2E latency, LLM/embedding provider, thumbs up/down.
* Dashboard (even in Streamlit): **daily hit@k**, estimated groundedness, average times, errors.

### 7) Acceptance criteria (PoC → Go/NoGo)

* Retrieval: **Recall@5 ≥ 0.80**, **MRR@10 ≥ 0.6**.
* Generation: **Groundedness ≥ 0.85**, **Contradiction ≈ 0%**, **Human score ≥ 4/5**.
* Operations: **p95 latency ≤ 4–6 s**, **error rate < 1%**.
* Memory: **MEM Coverage ≥ 30%**; no contradictory memory.

---

## Mini-howto (immediately applicable)

* **Gold set JSON**:

  ```json
  {
    "query_id": "Q123",
    "query": "VoIP one-way audio with external SBC",
    "answer_gold": "Disable SIP ALG and open RTP range.",
    "doc_gold": ["NETKB-3"],
    "mem_gold": []
  }
  ```
* **Eval script (offline)**:

  1. embed → retrieve → Recall/Hit/MRR/nDCG
  2. prompt → LLM → EM/F1/ROUGE
  3. groundedness (overlap/NLI)
  4. CSV with metrics + averages
* **A/B**: CLI flag for embedder, threshold, top-k, MEM cap.

**Tactical tips**

* **Chunking** long tickets: 200–400 tokens + 50 overlap → better recall.
* **Rerank** top-k (cosine + BM25 or MMR) → better MRR/nDCG.
* **Structured prompt** (steps, prerequisites, verification) → more actionable answers.
* **Different thresholds per source** (if adding PDFs): slightly stricter DOC threshold.

---

## UI Evolution

### A) Is Streamlit “professional” for a commercial product?

**Pros**: fast development, great for PoC/internal tools, easy deploy, many ready-made components.
**Cons**: no native multi-user/RBAC, complex UX harder, frequent reruns, limited branding.
**Recommendation**: Streamlit is fine for **internal teams/pilots**; for **SaaS/enterprise** consider classic web stack (React/Next + FastAPI, SSO, async ingestion workers, RAG as a service).

### B) Improve current UI while staying on Streamlit

* **Tabs**: “📥 Ingestion”, “🔎 Search”, “🧠 Playbook”, “⚙️ Config”.
* **Forms** for batch actions; **expanders** for advanced options; **status bar** with environment/provider/collection.
* **Usability**: disable buttons without prerequisites, clear toasts, copy-to-clipboard, useful spinners.
* **State & performance**: avoid `session_state` writes after widgets; `cache_data` / `cache_resource`; use forms to reduce reruns.
* **Tables**: pagination/playbook filters; similar results with clickable links and MEM expander.
* **Config & security**: clear sections in sidebar; masked API key; export/restore preferences.
* **i18n**: consider language switch if non-Italian users.
* **Product-grade migration**: signals—SSO, complex routing, async ingestion, external integrations.

**TL;DR**
Stay on Streamlit for current target: add **tabs + forms + status bar**, strengthen cache/state, refine **Playbook** tab (pagination, single delete, expander). Prepare a **thin FastAPI backend** for RAG logic as transition step.