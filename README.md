# YouTrack RAG Support App

Streamlit application for technical assistance based on YouTrack tickets, indexed in a local Vector DB (Chroma) and queried through retrieval‑augmented generation (RAG) using OpenAI or local Ollama LLMs.

---

## 1. Overview

This app lets you:

- Connect to a YouTrack instance via URL + Bearer token
- Load projects and issues, and index them into a Chroma vector store
- Interact directly with YouTrack via MCP (Model Context Protocol) for programmatic operations
- Upload and index PDF/DOCX/TXT documentation for CLI command reference
- Configure embeddings, chunking and retrieval behavior
- Choose an LLM provider (OpenAI or Ollama) and model
- Ask questions in natural language and get answers grounded on similar tickets or documentation
- Save good answers as reusable "playbooks" in a separate memory collection
- Persist non‑sensitive preferences locally across sessions.

The UI is organized as a **multi‑phase wizard** in the sidebar:

1. YouTrack connection
2. MCP Console
3. Embeddings & Vector DB
4. Retrieval configuration
5. LLM & API keys
6. Chat & Results
7. Solutions memory
8. Preferences & debug
9. Docs KB (PDF/DOCX/TXT)

---

## 2. Features by Phase

### 2.1 Phase 1 – YouTrack Connection

- Configure **YouTrack URL** and **Bearer token** (not saved to disk).  
- On “Connect”, the app creates a `YouTrackClient` and loads the list of projects.  
- A project selectbox shows entries as `Name (ShortName)`; when you select a project, issues are automatically loaded.  
- A “Reload issues” button lets you fetch them again manually.  
- Issues are shown in a Markdown table with:
  - Clickable **ID** linking back to YouTrack (`/issue/<ID>`)  
  - Shortened **Summary** on a single line.  

---

### 2.2 Phase 2 – MCP Console

The MCP Console provides direct programmatic access to YouTrack through the Model Context Protocol, allowing you to interact with your YouTrack instance without going through the RAG workflow.

#### Features

- **Direct YouTrack interaction**: Execute operations on YouTrack via natural language prompts
- **One-click prompt library**: 20+ preset prompts organized in 7 categories:
  - Quick status (project snapshots, recent updates)
  - Backlog & workload (state counts, assignee workload)
  - Aging & stuck (old issues, stale updates)
  - Critical & risks (blockers, SLA risks)
  - Workflow health (bottlenecks, regressions)
  - Trends (creation/resolution rates, lead time)
  - Planning (sprint candidates, meeting reports)
- **Project context**: Automatically insert the current project context from Phase 1
- **Auto-run toggle**: Execute preset prompts immediately on click
- **Test prompts**: Quick-start button to test MCP connectivity
- **Tabbed results**: View responses in three tabs:
  - **Readable**: Human-friendly formatted output
  - **Raw**: Complete response object for debugging
  - **Error**: Error messages if the request fails

#### How to use

1. **Connect to YouTrack** in Phase 1 (URL + Bearer token required)
2. **Configure OpenAI API key** in Phase 5 (MCP requires OpenAI)
3. **Navigate to MCP Console** (Phase 2)
4. **Option A - Use preset prompts:**
   - Select a category from the dropdown (e.g., "Quick status", "Critical & risks")
   - Click any preset button to load the prompt
   - Enable "Auto-run on click" to execute immediately
   - Presets automatically inject the current project key
5. **Option B - Write custom prompts:**
   - Write a natural language prompt, such as:
     - "Search the 10 most recent issues in the current project about 'VPN'"
     - "Get details for issue NETS-123"
     - "List all projects and their short names"
   - Click "Run MCP prompt" to execute

#### Use cases

- **Quick data exploration**: Search and inspect issues without indexing
- **Administrative tasks**: Query project metadata, issue counts, etc.
- **Integration testing**: Verify YouTrack connectivity and MCP server functionality
- **Ad-hoc queries**: Answer one-off questions without creating vector embeddings

**Note:** MCP operations are independent of the RAG workflow and do not affect or use the vector database.

---

### 2.3 Phase 3 – Embeddings & Vector DB

#### Chroma path and collections

- Configurable **Chroma path** (`persist_dir`), defaulting to:
  - `/tmp/chroma` in cloud / read‑only environments  
  - `<APP_DIR>/data/chroma` in local / Docker environments  
  - or a custom path via `CHROMA_DIR` env var / Streamlit secrets.
- The app lists existing Chroma collections and lets you:
  - Select an existing collection  
  - Or choose `➕ Create new collection…` and specify a name
- The selected collection name is stored as:
  - `collection_selected` / `vs_collection` in `session_state` and prefs.

#### Collection management

- **Delete collection** button:
  - Requires an explicit confirmation checkbox  
  - Deletes the Chroma collection  
  - Removes the associated `<collection>__meta.json` file  
  - Clears current issues, vector handle and related prefs  
  - Leaves you on Phase 2 after a rerun.  

#### Embeddings configuration

- Embedding providers:
  - `Local (sentence-transformers)` (when available and not in cloud)  
  - `OpenAI`  
- Embedding model options:
  - Local: `all-MiniLM-L6-v2`  
  - OpenAI: `text-embedding-3-small`, `text-embedding-3-large`  
- When you switch provider, the model is reset to a suitable default.  
- The chosen provider/model are used both for **indexing** and, unless overridden by metadata, for **query**.  

#### Ticket indexing (with chunking)

- “Index tickets” button indexes all currently loaded issues into the selected collection.  
- Long ticket texts are **chunked** with configurable parameters (see Phase 4):
  - Token‑based when `tiktoken` is available, otherwise whitespace‑based  
  - Metadata per chunk:
    - `parent_id` = original ticket ID  
    - `id_readable` = ticket ID  
    - `summary`, `project`  
    - `chunk_id`, `pos` (token offset) for multi‑chunk tickets.  
- The embedder input combines ID, summary and chunk text to improve semantic search.  
- After indexing:
  - A `<collection>__meta.json` file is written with `provider` and `model`  
  - The `vs_*` fields in `session_state` are updated (`vs_collection`, `vs_persist_dir`, `vs_count`)  
  - A success message with the total number of indexed chunks/documents is shown.  

---

### 2.4 Phase 4 – Retrieval Configuration

This phase controls how results are retrieved and aggregated from Chroma.

#### Distance threshold

- Slider `max_distance` (cosine distance), default **0.9**.
- Both KB (tickets) and MEM (playbooks) results are filtered: only those with `distance <= max_distance` are kept.

Typical usage:

- Lower values → more precise, fewer but highly relevant results
- Higher values → more permissive, useful when the KB is small or noisy

#### Chunking configuration

Controls how long tickets are split when indexing:

- `enable_chunking` (checkbox)
- `chunk_size` (tokens), default 800
- `chunk_overlap` (tokens), default 80
- `chunk_min`: below this size, tickets are indexed as a single document (default 512).

These settings are used in **Phase 3** during indexing via `split_into_chunks`.

#### Advanced retrieval settings

Under the “Advanced settings” expander:

- `show_distances`: show distance values next to results in the UI  
- `top_k`: number of KB results retrieved from Chroma (before filtering / collapsing)  
- `collapse_duplicates`: collapse multiple chunks from the same ticket in the UI  
- `per_parent_display`: max number of results per ticket shown in the UI  
- `per_parent_prompt`: max number of chunks per ticket used in the LLM prompt  
- `stitch_max_chars`: character limit when concatenating chunks into a single context block. 

There is also a **“Reset to defaults”** button that restores recommended values and shows a toast.

All these settings are synced to canonical keys used by the Chat phase (`top_k`, `show_distances`, `collapse_duplicates`, `per_parent_display`, `per_parent_prompt`, `stitch_max_chars`) and are persisted in prefs. 

---

### 2.5 Phase 5 – LLM & API Keys

- LLM providers:
  - **OpenAI**  
  - **Ollama (local)** – shown only if detected via HTTP `/api/tags` or `ollama list`  
- Provider change resets the model to:
  - `gpt-4o` for OpenAI  
  - `llama3.2` for Ollama (default)  
- Model is editable via a text input (`llm_model`).  
- Temperature slider between 0.0 and 1.5.  

**API Keys**

- The app determines whether an OpenAI key is needed based on:
  - Embeddings provider  
  - LLM provider  
- If needed, an “OpenAI API Key” password field is enabled.
- The key is kept in `session_state["openai_key"]`, never written to prefs.  

---

### 2.6 Phase 6 – Chat & Results

The core RAG workflow.

#### Query handling & embedder selection

- Uses the active `persist_dir` and `vs_collection` (or falls back to prefs / new collection name).  
- Ensures the vector collection is opened via `open_vector_in_session`.  
- For embeddings at query time:
  - Tries to read `<collection>__meta.json` (provider + model)  
  - If available, this overrides the current UI selection to ensure consistency  
  - If not, falls back to the embedding provider/model chosen in the UI.  
- Shows an info message if there is a mismatch between the embedding model used at index time and the one used at query time.

#### Retrieval from KB (tickets)

- Computes query embedding and runs `collection.query()` with `n_results = top_k`.  
- Filters results with `distance <= max_distance`.  
- Debug info shows:
  - raw number of results  
  - collection count  
  - first distances and threshold  

#### Retrieval from MEM (playbooks)

- If `enable_memory` is active:
  - Queries the separate `memories` collection  
  - Filters by distance threshold and TTL:
    - Only entries with `expires_at >= now` are kept  
  - Uses a cap `mem_cap = 2` to limit how many MEM items are blended.  

#### Blending KB + MEM and collapse logic

- MEM results (up to 2) are added first, then KB results until `top_k` total.  
- The combined list is processed twice via `collapse_by_parent`:
  - **View list**: `per_parent_display`, `stitch_for_prompt=False`  
  - **Prompt context**: `per_parent_prompt`, `stitch_for_prompt=True`, `stitch_max_chars` limit  
- Each group is built around `parent_id` / `id_readable` and sorted by distance and token position.  

#### Prompt and LLM answer

- System prompt (`RAG_SYSTEM_PROMPT`) instructs the model to:
  - Answer based on similar YouTrack tickets  
  - Always cite ticket IDs in brackets  
  - Ask for clarifications when context is insufficient  
  - Answer in **English**  
- The user prompt lists:
  - The new ticket text  
  - A summary of similar tickets with ID, distance, summary and first 500 characters  
- Optional “Show prompt” debug toggle displays the final prompt in an expander. 
- The answer is generated via `LLMBackend` using:
  - OpenAI Responses API (with fallback to Chat Completions)  
  - Or Ollama `/api/chat` with `stream=False` and robust JSON parsing fallback.  

#### Results display

- The final answer is shown at the top.  
- Below, a “Similar results (top‑k, with provenance)” section lists:
  - KB results:
    - Ticket ID + summary as a link back to YouTrack (when base URL is known)  
    - Optional distance and chunk information (ID, token offset)  
    - Chunk text in an expander  
  - MEM results:
    - Marked as `🧠 Playbook` with title (if present)  
    - Optional distance  
    - Optional full text if `mem_show_full` is enabled. 

---

### 2.7 Phase 7 – Solutions Memory

This page manages the **playbook memory** stored in the separate `memories` collection.

- Global toggle `enable_memory`:
  - Controls whether the Chat phase can save and retrieve playbooks  
- `mem_ttl_days`: default TTL (days) applied to new playbooks  
- `mem_show_full`: controls whether full playbook text is shown in Chat results  
- `show_memories`: enables the table of saved playbooks on this page.  

**Delete all memories**

- “Delete all memories” button:
  - Requires confirmation checkbox  
  - Deletes the `memories` collection and recreates it empty  

**Playbook table**

- When `show_memories` is enabled:
  - Reads all entries from `memories`  
  - Shows a dataframe with columns:
    - `ID`, `Project`, `Tags`, `Created`, `Expires`, `Preview` (short snippet).  

---

### 2.8 Phase 8 – Preferences & Debug

- Toggle **Enable preferences memory (local)**:
  - If enabled, non‑sensitive prefs are stored in `.app_prefs.json` (local or `/tmp` in cloud).  
- “Save preferences”:
  - Normalizes provider/model (e.g., forces OpenAI if Ollama is not available)  
  - Writes all relevant fields:
    - YouTrack URL  
    - persist_dir, collection names  
    - embedding backend/model  
    - LLM provider/model/temperature  
    - distance, chunking, advanced retrieval settings  
    - memory settings (TTL, show flags)  
- “Restore defaults”:
  - Deletes the prefs file and reruns Streamlit.  

**Debug**

- "Show LLM prompt" checkbox: same flag used by the Chat phase to optionally display the prompt.

---

### 2.9 Phase 9 – Docs KB (PDF/DOCX/TXT)

This phase provides a dedicated knowledge base for technical documentation (PDF/DOCX/TXT), separate from YouTrack tickets.

#### Features

- **Multi-format document upload**: PDF, DOCX, TXT via Streamlit file uploader
- **Robust text extraction**:
  - PDF: PyMuPDF → pdfplumber → PyPDF2 fallback chain
  - DOCX: python-docx paragraph extraction
  - TXT: UTF-8 with charset detection fallback (chardet)
- **Dedicated collection**: `docs_kb` ChromaDB collection (separate from tickets)
- **Document manifest**: SHA256-based deduplication with metadata tracking
- **RAG queries**: Specialized prompt for CLI documentation (Italian answers)
- **PDF export**: Export answers as formatted PDF with Markdown rendering
  - Supports headings, lists (ordered/unordered), code blocks, inline formatting
  - ReportLab-based structured document generation

#### How to use

1. **Navigate to Phase 9** (Docs KB)
2. **Upload documents**: Select PDF/DOCX/TXT files (multiple selection supported)
3. **Index documents**: Click "Index uploaded documents"
   - Extraction runs with fallback chain
   - Chunking applied (reuses Phase 4 settings)
   - Embeddings computed (uses Phase 5 embeddings provider)
   - Stored in `docs_kb` collection
4. **Ask questions**:
   - Enter question in Italian (e.g., "Come configuro RPKI? Fammi un esempio")
   - Select Top K (number of chunks to retrieve)
   - Enable/disable LLM answer
   - Click "Search in Docs KB"
5. **View results**:
   - Retrieved chunks shown in expander (with source file, chunk ID, distance)
   - LLM answer in Markdown with CLI examples in code blocks
   - Sources section at bottom
6. **Export to PDF** (optional):
   - Click "Generate PDF" to create downloadable PDF
   - Download via "Download answer as PDF" button

#### Document management

- **List indexed documents**: Shows filename, doc_id, chunks, size, timestamp
- **Delete documents**: Check "confirm" + click 🗑️ remove button
- **Deduplication**: SHA256 hash prevents re-indexing same file

#### Metadata preserved

Each document chunk stores:
- `doc_id`: SHA256 hash (first 16 chars)
- `source_file`: Original filename
- `doc_type`: pdf/docx/txt
- `chunk_id`: Chunk number (if multi-chunk)
- `pos`: Token offset (if chunking enabled)

#### Use cases

- **CLI command reference**: Index networking device CLI documentation
- **Configuration examples**: Search for specific command syntax
- **Troubleshooting guides**: Find procedures for common issues
- **Technical manuals**: Query device specifications, capabilities, limitations

**Note:** Docs KB operates independently from YouTrack. It uses the same embedding provider (Phase 5) and retrieval settings (Phase 4) but queries a separate collection.

---

## 3. Playbook Creation (Mark as Solved)

From the Chat page:

- If `enable_memory` is True and a last answer exists, you can press:  
  **“✅ Mark as solved → Save as playbook”**  
- The app:
  1. Builds a condensation prompt instructing the LLM to produce 3–6 imperative steps.  
  2. Calls the LLM (slightly lower temperature) to generate a compact playbook; on error, falls back to truncating the answer.  
  3. Builds metadata:
     - `source="memory"`, `project`, `quality="verified"`  
     - `created_at`, `expires_at = now + mem_ttl_days`  
     - `tags` including `playbook` and current project (if known)  
  4. Uses the current embedder to embed the playbook text and add it to `memories`.  
  5. Shows a caption with path, collection and count, and reopens the Solutions Memory page after rerun.  

---

## 4. Sidebar Wizard & Status Panels

The sidebar provides:

- Phase navigation (radio with 9 phases + progress bar)  
- YouTrack status (connected / not connected, current URL)  
- Vector DB / Embeddings summary:
  - persist_dir, active collection, embedding provider/model  
- LLM status:
  - provider, model, temperature  
- Retrieval summary (read‑only):
  - Top‑K, max distance, collapse duplicates  
  - Per‑ticket aggregation and stitch limit  
  - Chunking settings (enabled, size, overlap, min size)  
  - Embeddings + collection summary.  
- Embedding status:
  - “Indexed with” vs “Query using” (provider + model + metadata source)  
  - Warning if there is a mismatch between indexed and query settings  

On non‑cloud environments, a **Quit** button closes the app (`os._exit(0)`).  

The sidebar also automatically opens the active collection (if any) and shows the number of indexed documents.  

---

## 5. Requirements & Installation

### 5.1 Python dependencies

Install from `requirements.txt`, typically including:

- `streamlit`
- `chromadb`
- `sentence-transformers` (for local embeddings)
- `openai`
- `tiktoken` (optional, for token‑based chunking)
- `requests`, `pandas` and other standard utilities.

Additional dependencies for Phase 9 (Docs KB):
- **Document parsing**: `PyPDF2`, `python-docx`, `chardet`, `pymupdf`, `pdfplumber`
- **PDF export**: `reportlab`, `markdown`

```bash
pip install -r requirements.txt
```

### 5.2 Environment variables

Optional environment variables:

- `OPENAI_API_KEY` or `OPENAI_API_KEY_EXPERIMENTS`  
- `CHROMA_DIR` – overrides default Chroma path  
- `OLLAMA_HOST` – host/port for Ollama (default `http://localhost:11434`).  

---

## 6. Running the App

### 6.1 Quick Start (All-in-One)

**Recommended**: Start both Streamlit UI and FastAPI webhook service automatically:

```bash
# Python script (cross-platform)
python start_local.py

# Or bash script (Linux/Mac)
chmod +x start_local.sh
./start_local.sh
```

This starts:
- Streamlit UI on http://localhost:8502
- FastAPI webhook on http://127.0.0.1:8010
- API docs on http://127.0.0.1:8010/docs

Press `Ctrl+C` to stop both services.

### 6.2 Streamlit only (manual)

If you only need the UI without the webhook service:

```bash
streamlit run app.py --server.port 8502
```

Then open the browser at the URL printed by Streamlit.

### 6.3 FastAPI webhook only (manual)

If you need to run the webhook service separately:

```bash
uvicorn service_webhook:app --host 127.0.0.1 --port 8010 --log-level info
```

Access API docs at http://127.0.0.1:8010/docs

### 6.4 CLI self‑tests

If Streamlit is not available and you run:

```bash
python app.py
```

the app prints basic usage help and runs minimal self‑tests:

- VectorStore initialization
- Local embeddings (if `sentence-transformers` is installed)
- LLM backend initialization for OpenAI / Ollama (when possible).

---

## 7. Docker Deployment

### 7.1 Quick Start

```bash
# Copy environment template
cp .env.example .env
# Edit .env with your values

# Start both services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access services:
- Streamlit UI: http://localhost:8503
- FastAPI webhook: http://localhost:8010
- API docs: http://localhost:8010/docs

### 7.2 Configuration

The Docker setup automatically starts both Streamlit and FastAPI services. Configure via environment variables in `.env`:

**Required:**
```bash
OPENAI_API_KEY=sk-...
```

**Optional (webhook service):**
```bash
YT_BASE_URL=https://your-youtrack.myjetbrains.com
YT_TOKEN=perm:your-token-here
WEBHOOK_SECRET=your-secret-key
```

See `.env.example` for all available options.

### 7.3 Data Persistence

ChromaDB data is persisted in `./data_docker/chroma/` when using docker-compose.

```text
project-root/
    app.py
    service_webhook.py
    start_local.py
    start_docker.sh
    data/          ← local Chroma (when running on host)
    data_docker/   ← Chroma used inside Docker
```

### 7.4 Advanced Deployment

For production deployments, Streamlit Cloud limitations, separate FastAPI hosting, and troubleshooting, see [DEPLOYMENT.md](./DEPLOYMENT.md).

With this configuration:

- `APP_DIR` inside the container is `/app`  
- Default Chroma path becomes `/app/data/chroma`  
- Data is persisted under `./data_docker` on the host, separate from any local `./data`.  

If you get schema errors (e.g. from older local DBs), just remove `data_docker/chroma` and reindex.

---

## 8. License

See the `LICENSE` file if present in the repository.  
