# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application

**Streamlit Mode (Primary):**
```bash
streamlit run app.py --server.port 8502
```

**CLI Self-Tests:**
```bash
python app.py
```
Runs minimal self-tests for VectorStore, EmbeddingBackend, and LLMBackend without launching the UI.

### Docker

**Build and run:**
```bash
docker-compose up --build
```

**Access:** http://localhost:8503

**Important:** Docker uses separate data directory (`./data_docker`) from local development (`./data`).

### Dependencies

**Install:**
```bash
pip install -r requirements.txt
```

**Local development (with sentence-transformers):**
```bash
pip install -r requirements-local.txt
```

## Architecture Overview

### Application Structure

This is a **Streamlit-based RAG (Retrieval-Augmented Generation) system** organized as a 7-phase wizard:

1. **YouTrack Connection** (lines 760-850): Connects to YouTrack API, loads projects/issues
2. **Embeddings & Vector DB** (lines 852-1290): Configures Chroma collections, indexes tickets with chunking
3. **Retrieval Configuration** (lines 1292-1527): Distance threshold, chunking parameters, advanced retrieval settings
4. **LLM & API Keys** (lines 1528-1665): LLM provider/model selection (OpenAI/Ollama)
5. **Chat & Results** (lines 1667-2100): Main RAG workflow - query, retrieve, generate answer
6. **Solutions Memory** (lines 2101-2270): Manage reusable "playbooks" saved from solved tickets
7. **Preferences & Debug** (lines 2271-2407): Persist settings, debug controls

### Core Components

**YouTrackClient** (lines 449-487):
- REST API client for fetching projects and issues
- Uses Bearer token authentication
- Issues are fetched with limit=200 per project

**EmbeddingBackend** (lines 493-516):
- Abstraction over OpenAI and local sentence-transformers
- Supports: `text-embedding-3-small`, `text-embedding-3-large`, `all-MiniLM-L6-v2`
- Lazy initialization of models

**VectorStore** (lines 539-563):
- Wrapper around ChromaDB collections
- Uses cosine distance for similarity
- Collections persist in `persist_dir` (default: `<APP_DIR>/data/chroma` or `/tmp/chroma` in cloud)

**LLMBackend** (lines 580-675):
- Abstraction over OpenAI and Ollama
- OpenAI: tries Responses API, falls back to Chat Completions
- Ollama: HTTP REST via `/api/chat` with robust JSON parsing

### Key Technical Patterns

**Chunking Strategy** (lines 193-219):
- Long ticket descriptions are split using token-based sliding window
- Parameters: `chunk_size=800`, `chunk_overlap=80`, `chunk_min=512`
- Uses `tiktoken` when available, otherwise whitespace-based
- Each chunk maintains metadata: `parent_id`, `chunk_id`, `pos` (token offset)
- Purpose: Respects LLM context windows while preserving semantic granularity

**Result Collapsing & Stitching** (lines 699-733):
```python
collapse_by_parent(results, per_parent, stitch_for_prompt, max_chars)
```
- Groups chunks by `parent_id` (original ticket)
- Two usage modes:
  - **Display** (`per_parent=1`): Show top chunk per ticket in UI
  - **Prompting** (`per_parent=3`, `stitch_for_prompt=True`): Concatenate multiple chunks (up to 1500 chars) for richer LLM context
- Results are sorted by distance and token position

**Metadata Injection** (lines 1194-1206):
- After indexing, `<collection>__meta.json` records embedding provider/model
- During query (lines 1789-1803), metadata is read to ensure consistency
- Warning shown if query embeddings differ from indexed embeddings

**Solutions Memory (Playbooks)** (lines 2101-2250):
- Separate `memories` collection for reusable solutions
- TTL-based expiration (default 180 days)
- Condensed via LLM into 3-6 imperative steps
- Retrieved alongside KB results (up to 2 playbooks per query)
- Marked with 🧠 icon in UI

### Session State Management

**Three Layers:**

1. **Sticky Preferences** (`.app_prefs.json`):
   - Non-sensitive settings (YouTrack URL, Chroma path, model selections, retrieval params)
   - Persisted via `save_prefs()` / `load_prefs()` (lines 131-167)
   - **Never stores:** `yt_token`, `openai_key`

2. **Runtime State** (`session_state`):
   - Active clients: `yt_client`, `vector`, `embedder`
   - Loaded data: `projects`, `issues`
   - Current selections: `vs_collection`, `collection_selected`

3. **UI State**:
   - Widget values persisted with `key` parameter
   - Automatically restored across reruns

**Initialization:** `init_prefs_in_session()` loads prefs into session_state at startup

### Data Flow

```
YouTrack API → YTIssue objects
    ↓
split_into_chunks() → chunks with metadata
    ↓
EmbeddingBackend.embed() → vectors
    ↓
VectorStore.add() → Chroma collection
    ↓
User Query → EmbeddingBackend.embed()
    ↓
VectorStore.query() → similar chunks (filtered by distance threshold)
    ↓
collapse_by_parent() → grouped/stitched results
    ↓
build_prompt() + RAG_SYSTEM_PROMPT → LLM context
    ↓
LLMBackend.generate() → Answer with [TICKET-ID] citations
    ↓
(Optional) Save as playbook → memories collection
```

### Prompt Construction

**System Prompt** (lines 678-681):
```
You are a technical assistant who answers based on similar YouTrack tickets.
Always cite the IDs of the tickets found in square brackets.
If context is insufficient, ask for clarifications. Use english in the answer
```

**User Prompt** (lines 683-694):
```
New ticket:
<user query>

Similar tickets found (closest first):
- ID | distance=X.XXX | Summary
  <chunk text (first 500 chars)>
- ...
```

### Cloud Detection & Compatibility

**Cloud Detection** (lines 54-71):
- Attempts to write test file in app directory
- If fails → `IS_CLOUD = True` → use `/tmp` for Chroma and prefs
- Disables local models (sentence-transformers, Ollama) in cloud

**NumPy 2 Compatibility** (lines 41-50):
- ChromaDB has issues with NumPy 2.0+
- Code patches missing attributes: `np.float_`, `np.int_`, etc.

## Important Implementation Details

### Distance Thresholding
- Default `max_distance = 0.9` (cosine distance)
- Both KB (tickets) and MEM (playbooks) results filtered by this threshold
- Lower values → more precise, fewer results
- Higher values → more permissive, useful for small/noisy KB

### Collection Management
- **Two collections:** main KB (user-named) + `memories` (playbooks)
- **Delete collection:** requires confirmation checkbox, removes both Chroma collection and `<collection>__meta.json`
- **Automatic opening:** Sidebar attempts to open active collection and show document count

### Environment Variables
- `OPENAI_API_KEY` or `OPENAI_API_KEY_EXPERIMENTS` - OpenAI authentication
- `CHROMA_DIR` - Override default Chroma path
- `OLLAMA_HOST` - Ollama endpoint (default: `http://localhost:11434`)

### Bootstrap Scripts
Three CLI utilities in `/bootstraps/` for generating test data:
- `yt_nets_bootstrap.py`
- `yt_netkb_bootstrap.py`
- `yt_health_bootstrap.py`

### Docker Data Separation
- **Host data:** `./data/` (local development)
- **Docker data:** `./data_docker/` (container volume)
- Prevents schema conflicts between local and Docker ChromaDB instances

## Development Patterns

### Adding New LLM Providers
Extend `LLMBackend.__init__()` and `LLMBackend.generate()` (lines 580-675). Follow existing OpenAI/Ollama pattern with fallback logic.

### Modifying Retrieval Logic
- **Chunking:** Adjust `split_into_chunks()` (lines 193-219)
- **Collapsing:** Modify `collapse_by_parent()` (lines 699-733)
- **Prompting:** Edit `build_prompt()` (lines 683-694) or `RAG_SYSTEM_PROMPT` (lines 678-681)

### Adding New Phases
1. Create `render_phase_N()` function following existing pattern (lines 760-2407)
2. Update `PHASE_OPTIONS` dictionary at top of `run_streamlit_app()`
3. Add phase to sidebar navigation radio
4. Update progress bar logic

### State Persistence
To persist new settings:
1. Add to `DEFAULT_PREFS` dictionary
2. Add to `save_prefs()` (lines 131-167)
3. Add to `init_prefs_in_session()` bootstrap
4. **Never persist:** API keys, tokens, or sensitive credentials

## File Structure Reference

| Path | Purpose |
|------|---------|
| `app.py` | Main Streamlit application (single-file architecture) |
| `.app_prefs.json` | Persisted user preferences (local/`/tmp`) |
| `data/chroma/` | ChromaDB storage (local development) |
| `data_docker/chroma/` | ChromaDB storage (Docker container) |
| `.streamlit/secrets.toml` | Streamlit secrets (API keys, optional) |
| `requirements.txt` | Core dependencies (no local models) |
| `requirements-local.txt` | Local development (with sentence-transformers) |
| `docker-compose.yml` | Docker orchestration |
| `Dockerfile` | Container build configuration |
| `bootstraps/` | Test data generation scripts |

## Testing

**Manual Testing Flow:**
1. Run `streamlit run app.py --server.port 8502`
2. Phase 1: Connect with YouTrack URL + Bearer token
3. Phase 2: Select/create collection, index issues
4. Phase 3: Tune distance threshold (try 0.7-1.0 range)
5. Phase 4: Configure LLM (OpenAI requires API key)
6. Phase 5: Submit query, verify answer has [TICKET-ID] citations
7. Verify similar results show correct tickets with distances
8. (Optional) Save answer as playbook in Phase 6

**Self-Tests:**
```bash
python app.py
```
Validates VectorStore, EmbeddingBackend, and LLMBackend initialization.

## Known Issues & Robustness

**ChromaDB Schema Errors:**
- If you see schema errors, delete `data/chroma/` or `data_docker/chroma/` and reindex
- Caused by ChromaDB version upgrades or metadata changes

**Ollama Response Variability:**
- Code handles multiple JSON response formats (streaming, nested, various field names)
- See robust parsing in `LLMBackend.generate()` (lines 641-675)

**Embedding Model Mismatch:**
- Warning shown in sidebar if query embeddings differ from indexed embeddings
- Recommendation: reindex collection when changing embedding models

**NumPy 2 Compatibility:**
- Automatic patching applied at startup (lines 41-50)
- If issues persist, downgrade to NumPy 1.x: `pip install 'numpy<2'`
