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

This is a **Streamlit-based RAG (Retrieval-Augmented Generation) system** organized as an 8-phase wizard:

1. **YouTrack Connection**: Connects to YouTrack API, loads projects/issues
2. **MCP Console**: Direct programmatic access to YouTrack via Model Context Protocol
3. **Embeddings & Vector DB**: Configures Chroma collections, indexes tickets with chunking
4. **Retrieval Configuration**: Distance threshold, chunking parameters, advanced retrieval settings
5. **LLM & API Keys**: LLM provider/model selection (OpenAI/Ollama)
6. **Chat & Results**: Main RAG workflow - query, retrieve, generate answer
7. **Solutions Memory**: Manage reusable "playbooks" saved from solved tickets
8. **Preferences & Debug**: Persist settings, debug controls

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

**MCP Console** (lines 882-1015):
- Direct interaction with YouTrack via Model Context Protocol (MCP)
- Uses OpenAI Responses API with MCP tool integration
- Independent from RAG workflow - operates directly on YouTrack without vector DB
- Key function: `run_mcp_prompt(prompt, yt_url, yt_token, openai_key)` returns `{ok, readable, raw, error}`
- UI features: project context insertion, test prompts, tabbed results (Readable/Raw/Error)

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

### Using MCP Console
The MCP Console provides direct programmatic access to YouTrack without requiring vector indexing:

**Use Cases:**
- **Quick data exploration**: Search and inspect issues without indexing
- **Administrative queries**: Get project metadata, issue counts, custom field values
- **Testing connectivity**: Verify YouTrack API and MCP server functionality
- **Ad-hoc operations**: One-off queries that don't justify full RAG setup

**How it works:**
1. User writes natural language prompt (e.g., "Search 10 recent issues about VPN")
2. `run_mcp_prompt()` calls OpenAI Responses API with MCP tool configuration
3. OpenAI model decides when to call YouTrack MCP server tools
4. MCP server executes YouTrack API calls with Bearer token authentication
5. Results returned in three formats: readable (formatted), raw (full object), error (if failed)

**Important:** MCP operations are completely independent from the RAG workflow. They don't create embeddings, don't use the vector database, and don't affect indexed collections.

### Adding New LLM Providers
Extend `LLMBackend.__init__()` and `LLMBackend.generate()`. Follow existing OpenAI/Ollama pattern with fallback logic.

### Modifying Retrieval Logic
- **Chunking:** Adjust `split_into_chunks()`
- **Collapsing:** Modify `collapse_by_parent()`
- **Prompting:** Edit `build_prompt()` or `RAG_SYSTEM_PROMPT`

### Adding New Phases
1. Create `render_phase_N()` function following existing pattern
2. Update `PHASE_OPTIONS` and `PHASE_COLORS` dictionaries
3. Add phase icon to `PHASE_ICONS`
4. Add phase to sidebar navigation radio
5. Update progress bar logic

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
3. Phase 2 (MCP Console):
   - Click "Test MCP" button to verify connectivity
   - Try custom prompt: "Search the 5 most recent issues in the current project"
   - Verify tabbed results (Readable/Raw/Error) display correctly
4. Phase 3: Select/create collection, index issues
5. Phase 4: Tune distance threshold (try 0.7-1.0 range)
6. Phase 5: Configure LLM (OpenAI requires API key for RAG and MCP)
7. Phase 6: Submit query, verify answer has [TICKET-ID] citations
8. Verify similar results show correct tickets with distances
9. (Optional) Save answer as playbook in Phase 7

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

**MCP Console Requirements:**
- **Requires OpenAI API**: MCP Console only works with OpenAI (Responses API with MCP tools)
- **Requires YouTrack Bearer token**: Set in Phase 1
- **Independent operation**: MCP calls don't require vector DB or embeddings configuration
- **Error handling**: Check Error tab if MCP calls fail (common: missing credentials, network issues)
- **Server URL format**: MCP server endpoint is automatically constructed as `{yt_url}/mcp`
