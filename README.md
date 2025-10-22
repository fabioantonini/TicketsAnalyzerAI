# YouTrack RAG Support App

Applicazione Streamlit per assistenza tecnica basata su ticket YouTrack indicizzati in un Vector DB (Chroma) e consultati via retrieval‑augmented generation (RAG) usando LLM OpenAI o locali via Ollama.

---

## 🚀 Funzionalità principali

**Connessione a YouTrack**
- Login via URL + Token (Bearer), caricamento elenco progetti e **auto‑caricamento** dei ticket alla selezione del progetto.
- Tabella con ID, Summary, Project e link diretti all’issue su YouTrack.

**Gestione Vector DB (Chroma)**
- Configurazione del percorso di persistenza (Chroma path).
- **Selectbox delle collection** esistenti con opzione **“➕ Crea nuova collection…”** (fix sul naming: il nome digitato viene rispettato).
- **Apertura automatica** della collection selezionata (senza re‑indicizzazione forzata).
- **Eliminazione collection** dalla sidebar (checkbox di conferma + pulsante), con aggiornamento automatico della lista e rimozione del relativo file meta.

**Indicizzazione ticket**
- Indicizzazione dei ticket correnti nella collection selezionata, con embeddings scelti dall’utente.
- Salvataggio di un **file meta provider/modello** (`<collection>__meta.json`) per avvisare in query se il modello corrente differisce da quello usato in indicizzazione.
- Comportamento idempotente consigliato: `upsert` (se disponibile) oppure filtro degli ID già presenti prima di `add`.

**Embeddings**
- Provider selezionabile: **Locale (sentence‑transformers)** o **OpenAI**.
- **Modelli suggeriti automaticamente** in base al provider:
  - sentence‑transformers → `all-MiniLM-L6-v2`
  - OpenAI → `text-embedding-3-small`, `text-embedding-3-large`

**LLM**
- Provider selezionabile: **OpenAI** oppure **Ollama (locale)**.
- Selettore del modello LLM (reset automatico al cambio provider) e **slider “Temperature”** in sidebar.
- Gestione robusta delle risposte Ollama (`stream=False` + fallback per JSON concatenati) per evitare errori tipo “Extra data”.

**Chat RAG**
- Top‑k configurabile e **soglia massima di distanza** per filtrare i vicini: se nessun risultato è sotto soglia, la lista “Risultati simili” non viene mostrata.
- **Toggle “Mostra prompt LLM”** in sidebar: permette di visualizzare, in un expander, il prompt effettivamente inviato al modello.
- **Fix duplicazione output**: una sola generazione risposta e un’unica lista di risultati simili (cliccabili) quando presenti.
- **Provenienza chiara** dei risultati: `[KB]` (ticket indicizzati) e `[MEM]` (playbook della memoria). Link cliccabili agli issue YouTrack anche senza sessione attiva (fallback all’URL salvato).

**🧠 Memoria soluzioni – Playbook (Livello B)**
- Toggle **“Abilita memory ‘playbook’”** in sidebar.
- Bottone **“✅ Segna come risolto → Salva come playbook”** dopo la risposta: crea un mini‑playbook (3–6 frasi) riutilizzabile, con metadati (`project`, `quality`, `created_at`, `expires_at`/TTL, `tags`).
- Playbook salvati in **collection separata `memories`** (Chroma), **mai** mischiati al KB.
- In query: **retrieval combinato** KB ⊕ MEM (stessa soglia di distanza; cap delle MEM a 1‑2 risultati), con anteprima in linea e, opzionalmente, **expander** “Playbook completo” (attivabile dalla sidebar).
- Vista **“Playbook salvati”**: tabella con ID, progetto, tag, date e anteprima; pulsante per **cancellare tutte le memorie**.
- Debug non invasivo: caption con path e count; opzionale stampa delle distanze MEM.

**⚙️ Configurazione & preferenze (Sticky prefs – Livello A)**
- Salvataggio locale (file **`.app_prefs.json`** accanto a `app.py`) di **impostazioni non sensibili**: URL YouTrack, Chroma path, provider/modello embeddings, provider/modello LLM, temperatura, soglia distanza, selezione collection, toggle prompt, ecc.
- Pulsanti **“Salva preferenze”** e **“Ripristina default”**; ricarico immediato in sessione.
- Reset automatico dei **modelli** quando cambia il provider (LLM/Embeddings) per evitare inconsistenze.
- Protezioni: non salvare stringhe vuote (es. modello LLM); default sicuri al primo avvio; validazione a runtime.

**API Keys**
- **Override OpenAI API Key** dalla sidebar (textbox): prioritaria rispetto alle variabili d’ambiente; disabilitata automaticamente quando non necessaria (LLM=Ollama **e** Embeddings=Locali).

**Utility e UX**
- Pulsante **Quit** in sidebar (uscita pulita, senza shadowing di `os`).
- Layout **wide** e tabelle compatte con link diretti ai ticket.

---

## 📦 Librerie utilizzate

- [streamlit](https://streamlit.io/) – interfaccia grafica
- [requests](https://docs.python-requests.org/) – connessione a YouTrack e API esterne
- [pandas](https://pandas.pydata.org/) – gestione tabelle
- [chromadb](https://docs.trychroma.com/) – database vettoriale persistente
- [sentence-transformers](https://www.sbert.net/) – embeddings locali
- [openai](https://github.com/openai/openai-python) – embeddings e LLM in cloud
- [ollama](https://ollama.ai/) – esecuzione modelli LLM locali

---

## 🧰 Requisiti e installazione

### ⚙️ Dipendenze
- `streamlit`, `requests`, `chromadb`, `sentence-transformers`, `openai`, `tiktoken`, `pandas` (Ollama opzionale per LLM locali).

### ▶️ Avvio
```bash
streamlit run app.py --server.port 8502
```
Modalità CLI (fallback): senza Streamlit, il file espone self‑tests minimi eseguibili da terminale.

---

## ⚡️ Configurazione rapida

1. Inserisci in sidebar l’URL e il Token (Bearer) di YouTrack.
2. Seleziona/crea una **collection** nel Vector DB (Chroma) e verifica l’apertura automatica.
3. Scegli **Embeddings** e **LLM** (OpenAI o Ollama). Se necessario, inserisci la **OpenAI API Key** in sidebar.
4. Carica un progetto: i ticket vengono **auto‑caricati** e mostrati in tabella.
5. Premi **Indicizza ticket** per popolare la collection (prima volta o aggiornamento).
6. Vai alla sezione **Chatbot RAG**: inserisci il testo del nuovo ticket e premi **Cerca e rispondi**.
   - Opzionale: abilita **Mostra prompt LLM** per vedere il prompt finale.
   - Opzionale: abilita **Memoria playbook** e salva le soluzioni verificate.
   - Regola la **soglia distanza** per filtrare i risultati simili mostrati (vale anche per MEM).

---

## 🔒 Sicurezza

**Credenziali**
- **OpenAI API Key**: può essere fornita da variabile d’ambiente o inserita in **sidebar**; la chiave UI **non viene salvata su disco** e resta nello `st.session_state`.
- **YouTrack Token**: inserito in sidebar, usato solo per le chiamate API e **non** scritto su file.

**Dati e log**
- I ticket vengono indicizzati **localmente** in Chroma (path configurabile). Evita di condividere il datastore se contiene dati sensibili.
- La visualizzazione del prompt è **opt‑in** (disattivata di default). Evita di includere dati sensibili nel prompt.

**Memoria (Playbook)**
- È **opt‑in** (toggle in sidebar) e **separata** dal KB (collection `memories`). Non salvare PII o segreti nei playbook.
- Ogni playbook ha un **TTL** configurabile; disponibile pulsante “Elimina tutte le memorie”.

**LLM locali vs cloud**
- Con **Ollama** i dati **non** lasciano la macchina.
- Con **OpenAI**, prompt e contesti vengono inviati al servizio cloud del provider scelto.

---

## 📈 Stato attuale

**Completato**
- Connessione a YouTrack, selezione progetto e **auto‑caricamento** ticket con tabella e link.
- Gestione Chroma: selezione/creazione/eliminazione collection, **apertura automatica** senza re‑indicizzazione obbligatoria (fix del naming + rimozione meta).
- Indicizzazione con scelta embeddings; **meta provider/modello** e avviso in caso di mismatch durante la query.
- Chat RAG con **soglia distanza**, **visualizzazione prompt** a richiesta e **temperature** per LLM.
- Gestione robusta Ollama (no JSON concatenati).
- **Override OpenAI API Key** dalla sidebar con abilitazione/disabilitazione dinamica.
- **Sticky prefs (Livello A)**: salvataggio e ripristino di impostazioni non sensibili, reset automatico modelli al cambio provider.
- **Memoria Playbook (Livello B)**: salvataggio playbook, retrieval combinato KB⊕MEM, provenienza chiara, vista tabellare con gestione e cancellazione.

**In corso / raccomandazioni**
- **Deduplica/upsert** su Chroma per aggiornare ticket già indicizzati.
- **Chunking / Re‑rank / MMR**: da valutare per descrizioni lunghe o risultati troppo omogenei.
- Allerta quando l’**embedder delle MEM** differisce da quello corrente (per distanze più stabili).

---

## 🛠️ Troubleshooting

- **Ollama – “Extra data: line … column …”** → assicurarsi che l’app usi `stream=False` e che l’endpoint risponda con JSON singolo.
- **Nessun risultato simile** → alzare (numericamente) la **soglia distanza**; ricordare che ora vale anche per le MEM.
- **Playbook non compare tra i risultati** → verificare: (1) toggle **Memoria playbook** attivo, (2) **persist_dir** coerente tra salvataggio e lettura, (3) soglia distanza non troppo stretta, (4) embedder coerente (stesso modello di embeddings).
- **Checkbox “Mostra playbook salvati” che “sfarfalla”** → assicurarsi che ci sia **una sola key** (`show_memories`) e, dopo il salvataggio, usare il pattern *set on next run* (`open_memories_after_save` + `st.rerun()`).
- **Modello LLM vuoto** → la UI ora impedisce di salvare valori vuoti e valida prima della chiamata; se modificato il file prefs a mano, reimpostare dalla sidebar.

---

## 🗂️ Struttura progetto

- `app.py` — applicazione Streamlit completa (UI, ingestion, retrieval, memory, chat).
- `data/chroma/` — persistenza Chroma (inclusi file meta `collection__meta.json` e collection `memories`).
- `.app_prefs.json` — preferenze non sensibili (accanto a `app.py`).
- `README.md` — questo documento.

---

## 📄 Licenza

Vedi file LICENSE se presente nel repository.

---

## 🌱 Opzioni future

### LLM da Hugging Face
- **Server OpenAI‑compatible (consigliato)**: esponi il modello HF tramite un server compatibile con le API OpenAI (es. vLLM/TGI). La tua app può riusare lo stesso backend OpenAI puntando a `base_url` custom.
  ```bash
  # esempio avvio vLLM in locale
  pip install vllm
  python -m vllm.entrypoints.openai.api_server     --model meta-llama/Meta-Llama-3.1-8B-Instruct     --host 0.0.0.0 --port 8000
  # poi usa base_url: http://localhost:8000/v1
  ```
- **Transformers in‑process (alternativa)**: carica il modello direttamente in app con `transformers` (CPU/GPU locale). Pro: zero servizi esterni. Contro: warm‑up più lento e uso RAM/VRAM elevato.
