# YouTrack RAG Support App

Streamlit application for technical assistance based on YouTrack tickets indexed in a Vector DB (Chroma) and queried via retrieval‑augmented generation (RAG) using OpenAI or local Ollama LLMs.

---

## 🚀 Main Features

**YouTrack Connection**

* Login via URL + Token (Bearer), load the project list and **auto‑load** tickets when selecting a project.
* Table with ID, Summary, Project, and direct links to issues on YouTrack.

**Vector DB Management (Chroma)**

* Configure persistence path (Chroma path).
* **Selectbox of existing collections** with **“➕ Create new collection…”** option (naming fix: entered name is respected).
* **Automatic opening** of the selected collection (no forced re‑indexing).
* **Delete collection** from sidebar (confirmation checkbox + button), with automatic list refresh and removal of the related meta file.

**Ticket Indexing**

* Index current tickets into the selected collection using user‑chosen embeddings.
* Save a **meta provider/model file** (`<collection>__meta.json`) to warn during query if the current model differs from the one used in indexing.
* Recommended idempotent behavior: `upsert` (if available) or filtering existing IDs before `add`.

**Embeddings**

* Selectable provider: **Local (sentence‑transformers)** or **OpenAI**.
* **Automatically suggested models** based on provider:

  * sentence‑transformers → `all-MiniLM-L6-v2`
  * OpenAI → `text-embedding-3-small`, `text-embedding-3-large`

**LLM**

* Selectable provider: **OpenAI** or **Ollama (local)**.
* LLM model selector (automatic reset when provider changes) and **“Temperature” slider** in the sidebar.
* Robust Ollama response handling (`stream=False` + fallback for concatenated JSON) to avoid “Extra data” errors.

**RAG Chat**

* Configurable Top‑k and **maximum distance threshold** to filter neighbors: if no result is below threshold, the “Similar results” list is hidden.
* **“Show LLM Prompt” toggle** in sidebar: displays, in an expander, the actual prompt sent to the model.
* **Duplicate output fix**: single answer generation and single list of similar results (clickable) when available.
* **Clear provenance** of results: `[KB]` (indexed tickets) and `[MEM]` (memory playbooks). Clickable links to YouTrack issues even without active session (fallback to saved URL).

**🧠 Solution Memory – Playbook (Level B)**

* **“Enable playbook memory”** toggle in sidebar.
* **“✅ Mark as solved → Save as playbook”** button after an answer: creates a reusable mini‑playbook (3–6 sentences) with metadata (`project`, `quality`, `created_at`, `expires_at`/TTL, `tags`).
* Playbooks saved in a **separate `memories` collection** (Chroma), **never** mixed with KB.
* In query: **combined retrieval** KB ⊕ MEM (same distance threshold; MEM cap 1‑2 results), with inline preview and optional **“Full playbook” expander** (sidebar toggle).
* **“Saved Playbooks”** view: table with ID, project, tags, dates, and preview; button to **delete all memories**.
* Non‑intrusive debug: caption with path and count; optional MEM distance display.

**⚙️ Configuration & Preferences (Sticky prefs – Level A)**

* Local storage (file **`.app_prefs.json`** next to `app.py`) of **non‑sensitive settings**: YouTrack URL, Chroma path, embedding provider/model, LLM provider/model, temperature, distance threshold, collection selection, prompt toggle, etc.
* **“Save preferences”** and **“Restore defaults”** buttons; immediate session reload.
* Automatic model reset when provider changes (LLM/Embeddings) to avoid inconsistencies.
* Protections: do not save empty strings (e.g. empty LLM model); safe defaults at first launch; runtime validation.

**API Keys**

* **Override OpenAI API Key** from sidebar (textbox): takes priority over environment variables; automatically disabled when not needed (LLM=Ollama **and** Embeddings=Local).

**Utility and UX**

* **Quit** button in sidebar (clean exit, no `os` shadowing).
* **Wide layout** and compact tables with direct links to tickets.

---

## 📦 Libraries Used

* [streamlit](https://streamlit.io/) – graphical interface
* [requests](https://docs.python-requests.org/) – connection to YouTrack and external APIs
* [pandas](https://pandas.pydata.org/) – table handling
* [chromadb](https://docs.trychroma.com/) – persistent vector database
* [sentence-transformers](https://www.sbert.net/) – local embeddings
* [openai](https://github.com/openai/openai-python) – embeddings and cloud LLM
* [ollama](https://ollama.ai/) – execution of local LLMs

---

## 🧰 Requirements and Installation

### ⚙️ Dependencies

* `streamlit`, `requests`, `chromadb`, `sentence-transformers`, `openai`, `tiktoken`, `pandas` (Ollama optional for local LLMs).

### ▶️ Run

```bash
streamlit run app.py --server.port 8502
```

CLI mode (fallback): without Streamlit, the file exposes minimal self‑tests runnable from terminal.

---

## ⚡️ Quick Configuration

1. Enter YouTrack URL and Token (Bearer) in the sidebar.
2. Select/create a **collection** in the Vector DB (Chroma) and verify automatic opening.
3. Choose **Embeddings** and **LLM** (OpenAI or Ollama). If needed, enter the **OpenAI API Key** in the sidebar.
4. Load a project: tickets are **auto‑loaded** and displayed in a table.
5. Click **Index tickets** to populate the collection (first time or update).
6. Go to **RAG Chatbot** section: enter the text of the new ticket and press **Search and answer**.

   * Optional: enable **Show LLM Prompt** to view the final prompt.
   * Optional: enable **Playbook Memory** and save verified solutions.
   * Adjust **distance threshold** to filter similar results (applies to MEM too).

---

## 🔒 Security

**Credentials**

* **OpenAI API Key**: can be provided via environment variable or entered in **sidebar**; UI key **is not saved to disk** and remains in `st.session_state`.
* **YouTrack Token**: entered in sidebar, used only for API calls, and **not** written to file.

**Data and logs**

* Tickets are indexed **locally** in Chroma (configurable path). Avoid sharing datastore if it contains sensitive data.
* Prompt display is **opt‑in** (off by default). Avoid including sensitive data in the prompt.

**Memory (Playbook)**

* **Opt‑in** (toggle in sidebar) and **separate** from KB (collection `memories`). Do not save PII or secrets in playbooks.
* Each playbook has a **configurable TTL**; “Delete all memories” button available.

**Local vs Cloud LLMs**

* With **Ollama**, data **never leaves the machine**.
* With **OpenAI**, prompts and contexts are sent to the provider’s cloud service.

---

## 📈 Current Status

**Completed**

* YouTrack connection, project selection and **auto‑loading** tickets with table and links.
* Chroma management: collection selection/creation/deletion, **automatic opening** without mandatory re‑indexing (naming + meta removal fixes).
* Indexing with embedding selection; **meta provider/model** and warning if mismatch during query.
* RAG Chat with **distance threshold**, **prompt display**, and **temperature** for LLM.
* Robust Ollama management (no concatenated JSON).
* **OpenAI API Key override** from sidebar with dynamic enable/disable.
* **Sticky prefs (Level A)**: save/restore non‑sensitive settings, automatic model reset on provider change.
* **Playbook Memory (Level B)**: playbook saving, combined KB⊕MEM retrieval, clear provenance, tabular view with management and deletion.

**Ongoing / Recommendations**

* **Deduplication/upsert** on Chroma to update already indexed tickets.
* **Chunking / Re‑rank / MMR**: to be evaluated for long descriptions or overly homogeneous results.
* Alert when **MEM embedder** differs from current one (for more stable distances).

---

## 🛠️ Troubleshooting

* **Ollama – “Extra data: line … column …”** → ensure the app uses `stream=False` and the endpoint responds with single JSON.
* **No similar results** → increase (numerically) the **distance threshold**; remember it now applies to MEM too.
* **Playbook not shown among results** → check: (1) **Playbook Memory** toggle active, (2) consistent **persist_dir** between save and read, (3) distance threshold not too strict, (4) consistent embedder (same embedding model).
* **“Show saved playbooks” checkbox flickers** → ensure there’s **only one key** (`show_memories`); after saving, use *set on next run* pattern (`open_memories_after_save` + `st.rerun()`).
* **Empty LLM model** → UI now prevents saving empty values and validates before call; if the prefs file was manually edited, reconfigure from sidebar.

---

## 🗂️ Project Structure

* `app.py` — complete Streamlit app (UI, ingestion, retrieval, memory, chat).
* `data/chroma/` — Chroma persistence (including `collection__meta.json` meta files and `memories` collection).
* `.app_prefs.json` — non‑sensitive preferences (next to `app.py`).
* `README.md` — this document.

---

## 📄 License

See LICENSE file if present in the repository.

---

## 🌱 Future Options

### LLM from Hugging Face

* **OpenAI‑compatible server (recommended)**: expose HF model via OpenAI‑compatible API (e.g. vLLM/TGI). The app can reuse the same OpenAI backend by pointing to a custom `base_url`.

  ```bash
  # example local vLLM startup
  pip install vllm
  python -m vllm.entrypoints.openai.api_server     --model meta-llama/Meta-Llama-3.1-8B-Instruct     --host 0.0.0.0 --port 8000
  # then use base_url: http://localhost:8000/v1
  ```
* **In‑process Transformers (alternative)**: load the model directly in app with `transformers` (local CPU/GPU). Pros: no external services. Cons: slower warm‑up and higher RAM/VRAM usage.
