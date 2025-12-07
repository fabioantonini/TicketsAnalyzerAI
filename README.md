# YouTrack RAG Support App

Streamlit application for technical assistance based on YouTrack tickets, indexed in a local Vector DB (Chroma) and queried through retrieval‑augmented generation (RAG) using OpenAI or local Ollama LLMs.fileciteturn1file0

---

## 1. Overview

This app lets you:

- Connect to a YouTrack instance via URL + Bearer token  
- Load projects and issues, and index them into a Chroma vector store  
- Configure embeddings, chunking and retrieval behavior  
- Choose an LLM provider (OpenAI or Ollama) and model  
- Ask questions in natural language and get answers grounded on similar tickets  
- Save good answers as reusable “playbooks” in a separate memory collection  
- Persist non‑sensitive preferences locally across sessionsfileciteturn1file0

The UI is organized as a **multi‑phase wizard** in the sidebar:

1. YouTrack connection  
2. Embeddings & Vector DB  
3. Retrieval configuration  
4. LLM & API keys  
5. Solutions memory  
6. Chat & Results  
7. Preferences & debugfileciteturn1file0

---

## 2. Features by Phase

### 2.1 Phase 1 – YouTrack Connection

- Configure **YouTrack URL** and **Bearer token** (not saved to disk).  
- On “Connect”, the app creates a `YouTrackClient` and loads the list of projects.  
- A project selectbox shows entries as `Name (ShortName)`; when you select a project, issues are automatically loaded.  
- A “Reload issues” button lets you fetch them again manually.  
- Issues are shown in a Markdown table with:
  - Clickable **ID** linking back to YouTrack (`/issue/<ID>`)  
  - Shortened **Summary** on a single linefileciteturn1file0  

---

### 2.2 Phase 2 – Embeddings & Vector DB

#### Chroma path and collections

- Configurable **Chroma path** (`persist_dir`), defaulting to:
  - `/tmp/chroma` in cloud / read‑only environments  
  - `<APP_DIR>/data/chroma` in local / Docker environments  
  - or a custom path via `CHROMA_DIR` env var / Streamlit secretsfileciteturn1file0
- The app lists existing Chroma collections and lets you:
  - Select an existing collection  
  - Or choose `➕ Create new collection…` and specify a name
- The selected collection name is stored as:
  - `collection_selected` / `vs_collection` in `session_state` and prefs.fileciteturn1file0

#### Collection management

- **Delete collection** button:
  - Requires an explicit confirmation checkbox  
  - Deletes the Chroma collection  
  - Removes the associated `<collection>__meta.json` file  
  - Clears current issues, vector handle and related prefs  
  - Leaves you on Phase 2 after a rerunfileciteturn1file0  

#### Embeddings configuration

- Embedding providers:
  - `Local (sentence-transformers)` (when available and not in cloud)  
  - `OpenAI`  
- Embedding model options:
  - Local: `all-MiniLM-L6-v2`  
  - OpenAI: `text-embedding-3-small`, `text-embedding-3-large`  
- When you switch provider, the model is reset to a suitable default.  
- The chosen provider/model are used both for **indexing** and, unless overridden by metadata, for **query**.fileciteturn1file0  

#### Ticket indexing (with chunking)

- “Index tickets” button indexes all currently loaded issues into the selected collection.  
- Long ticket texts are **chunked** with configurable parameters (see Phase 3):
  - Token‑based when `tiktoken` is available, otherwise whitespace‑based  
  - Metadata per chunk:
    - `parent_id` = original ticket ID  
    - `id_readable` = ticket ID  
    - `summary`, `project`  
    - `chunk_id`, `pos` (token offset) for multi‑chunk ticketsfileciteturn1file0  
- The embedder input combines ID, summary and chunk text to improve semantic search.  
- After indexing:
  - A `<collection>__meta.json` file is written with `provider` and `model`  
  - The `vs_*` fields in `session_state` are updated (`vs_collection`, `vs_persist_dir`, `vs_count`)  
  - A success message with the total number of indexed chunks/documents is shownfileciteturn1file0  

---

### 2.3 Phase 3 – Retrieval Configuration

This phase controls how results are retrieved and aggregated from Chroma.

#### Distance threshold

- Slider `max_distance` (cosine distance), default **0.9**.  
- Both KB (tickets) and MEM (playbooks) results are filtered: only those with `distance <= max_distance` are kept.citeturn1file0  

Typical usage:

- Lower values → more precise, fewer but highly relevant results  
- Higher values → more permissive, useful when the KB is small or noisy  

#### Chunking configuration

Controls how long tickets are split when indexing:

- `enable_chunking` (checkbox)  
- `chunk_size` (tokens), default 800  
- `chunk_overlap` (tokens), default 80  
- `chunk_min`: below this size, tickets are indexed as a single document (default 512)citeturn1file0  

These settings are used in **Phase 2** during indexing via `split_into_chunks`.

#### Advanced retrieval settings

Under the “Advanced settings” expander:

- `show_distances`: show distance values next to results in the UI  
- `top_k`: number of KB results retrieved from Chroma (before filtering / collapsing)  
- `collapse_duplicates`: collapse multiple chunks from the same ticket in the UI  
- `per_parent_display`: max number of results per ticket shown in the UI  
- `per_parent_prompt`: max number of chunks per ticket used in the LLM prompt  
- `stitch_max_chars`: character limit when concatenating chunks into a single context blockciteturn1file0  

There is also a **“Reset to defaults”** button that restores recommended values and shows a toast.

All these settings are synced to canonical keys used by the Chat phase (`top_k`, `show_distances`, `collapse_duplicates`, `per_parent_display`, `per_parent_prompt`, `stitch_max_chars`) and are persisted in prefs.citeturn1file0  

---

### 2.4 Phase 4 – LLM & API Keys

- LLM providers:
  - **OpenAI**  
  - **Ollama (local)** – shown only if detected via HTTP `/api/tags` or `ollama list`citeturn1file0  
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
- The key is kept in `session_state["openai_key"]`, never written to prefs.citeturn1file0  

---

### 2.5 Phase 5 – Chat & Results

The core RAG workflow.

#### Query handling & embedder selection

- Uses the active `persist_dir` and `vs_collection` (or falls back to prefs / new collection name).  
- Ensures the vector collection is opened via `open_vector_in_session`.  
- For embeddings at query time:
  - Tries to read `<collection>__meta.json` (provider + model)  
  - If available, this overrides the current UI selection to ensure consistency  
  - If not, falls back to the embedding provider/model chosen in the UIciteturn1file0  
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
  - Uses a cap `mem_cap = 2` to limit how many MEM items are blended.citeturn1file0  

#### Blending KB + MEM and collapse logic

- MEM results (up to 2) are added first, then KB results until `top_k` total.  
- The combined list is processed twice via `collapse_by_parent`:
  - **View list**: `per_parent_display`, `stitch_for_prompt=False`  
  - **Prompt context**: `per_parent_prompt`, `stitch_for_prompt=True`, `stitch_max_chars` limit  
- Each group is built around `parent_id` / `id_readable` and sorted by distance and token position.citeturn1file0  

#### Prompt and LLM answer

- System prompt (`RAG_SYSTEM_PROMPT`) instructs the model to:
  - Answer based on similar YouTrack tickets  
  - Always cite ticket IDs in brackets  
  - Ask for clarifications when context is insufficient  
  - Answer in **English**  
- The user prompt lists:
  - The new ticket text  
  - A summary of similar tickets with ID, distance, summary and first 500 characters  
- Optional “Show prompt” debug toggle displays the final prompt in an expander.citeturn1file0  
- The answer is generated via `LLMBackend` using:
  - OpenAI Responses API (with fallback to Chat Completions)  
  - Or Ollama `/api/chat` with `stream=False` and robust JSON parsing fallback.citeturn1file0  

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
    - Optional full text if `mem_show_full` is enabledciteturn1file0  

---

### 2.6 Phase 6 – Solutions Memory

This page manages the **playbook memory** stored in the separate `memories` collection.

- Global toggle `enable_memory`:
  - Controls whether the Chat phase can save and retrieve playbooks  
- `mem_ttl_days`: default TTL (days) applied to new playbooks  
- `mem_show_full`: controls whether full playbook text is shown in Chat results  
- `show_memories`: enables the table of saved playbooks on this pageciteturn1file0  

**Delete all memories**

- “Delete all memories” button:
  - Requires confirmation checkbox  
  - Deletes the `memories` collection and recreates it empty  

**Playbook table**

- When `show_memories` is enabled:
  - Reads all entries from `memories`  
  - Shows a dataframe with columns:
    - `ID`, `Project`, `Tags`, `Created`, `Expires`, `Preview` (short snippet)citeturn1file0  

---

### 2.7 Phase 7 – Preferences & Debug

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

- “Show LLM prompt” checkbox: same flag used by the Chat phase to optionally display the prompt.citeturn1file0  

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
  5. Shows a caption with path, collection and count, and reopens the Solutions Memory page after rerun.citeturn1file0  

---

## 4. Sidebar Wizard & Status Panels

The sidebar provides:

- Phase navigation (radio with 7 phases + progress bar)  
- YouTrack status (connected / not connected, current URL)  
- Vector DB / Embeddings summary:
  - persist_dir, active collection, embedding provider/model  
- LLM status:
  - provider, model, temperature  
- Retrieval summary (read‑only):
  - Top‑K, max distance, collapse duplicates  
  - Per‑ticket aggregation and stitch limit  
  - Chunking settings (enabled, size, overlap, min size)  
  - Embeddings + collection summaryciteturn1file0  
- Embedding status:
  - “Indexed with” vs “Query using” (provider + model + metadata source)  
  - Warning if there is a mismatch between indexed and query settings  

On non‑cloud environments, a **Quit** button closes the app (`os._exit(0)`).citeturn1file0  

The sidebar also automatically opens the active collection (if any) and shows the number of indexed documents.citeturn1file0  

---

## 5. Requirements & Installation

### 5.1 Python dependencies

Install from `requirements.txt`, typically including:

- `streamlit`  
- `chromadb`  
- `sentence-transformers` (for local embeddings)  
- `openai`  
- `tiktoken` (optional, for token‑based chunking)  
- `requests`, `pandas` and other standard utilitiesciteturn1file0  

```bash
pip install -r requirements.txt
```

### 5.2 Environment variables

Optional environment variables:

- `OPENAI_API_KEY` or `OPENAI_API_KEY_EXPERIMENTS`  
- `CHROMA_DIR` – overrides default Chroma path  
- `OLLAMA_HOST` – host/port for Ollama (default `http://localhost:11434`)citeturn1file0  

---

## 6. Running the App

### 6.1 Streamlit mode (recommended)

```bash
streamlit run app.py --server.port 8502
```

Then open the browser at the URL printed by Streamlit.

### 6.2 CLI self‑tests

If Streamlit is not available and you run:

```bash
python app.py
```

the app prints basic usage help and runs minimal self‑tests:

- VectorStore initialization  
- Local embeddings (if `sentence-transformers` is installed)  
- LLM backend initialization for OpenAI / Ollama (when possible)citeturn1file0  

---

## 7. Docker Notes

The app is Docker‑friendly but does not enforce any specific volume layout.  
A practical pattern is:

```text
project-root/
    app.py
    data/          ← local Chroma (when running on host)
    data_docker/   ← Chroma used inside Docker
```

Example `docker-compose.yml`:

```yaml
services:
  rag-support-app:
    build: .
    container_name: rag_support_app
    ports:
      - "8503:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_PORT=8501
    volumes:
      - ./data_docker:/app/data
      - ./.streamlit:/app/.streamlit:ro
    restart: unless-stopped
```

With this configuration:

- `APP_DIR` inside the container is `/app`  
- Default Chroma path becomes `/app/data/chroma`  
- Data is persisted under `./data_docker` on the host, separate from any local `./data`.citeturn1file0  

If you get schema errors (e.g. from older local DBs), just remove `data_docker/chroma` and reindex.

---

## 8. License

See the `LICENSE` file if present in the repository.  
