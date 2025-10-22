# YouTrack RAG Support App

Applicazione Streamlit per assistenza tecnica basata su ticket YouTrack indicizzati in un Vector DB (Chroma) e consultati via retrieval‑augmented generation (RAG) usando LLM OpenAI o locali via Ollama.

---

## 🚀 Funzionalità principali

Connessione a YouTrack
- Login via URL + Token (Bearer), caricamento elenco progetti e **auto‑caricamento** dei ticket alla selezione del progetto.
- Tabella con ID, Summary, Project e link diretti all’issue su YouTrack.

Gestione Vector DB (Chroma)
- Configurazione del percorso di persistenza (Chroma path).
- **Selectbox delle collection** esistenti con opzione **“➕ Crea nuova collection…”** (fix sul naming: il nome digitato viene rispettato).
- **Apertura automatica** della collection selezionata (senza re‑indicizzazione forzata).
- **Eliminazione collection** dalla sidebar (checkbox di conferma + pulsante), con aggiornamento automatico della lista.

Indicizzazione ticket
- Indicizzazione dei ticket correnti nella collection selezionata, con embeddings scelti dall’utente.
- Salvataggio di un **file meta provider/modello** (collection__meta.json) per avvisare in query se il modello corrente differisce da quello usato in indicizzazione.
- Comportamento idempotente consigliato: `upsert` (se disponibile) oppure filtro degli ID già presenti prima di `add`.

Embeddings
- Provider selezionabile: **Locale (sentence‑transformers)** o **OpenAI**.
- **Modelli suggeriti automaticamente** in base al provider:
  - sentence‑transformers → `all-MiniLM-L6-v2`
  - OpenAI → `text-embedding-3-small`, `text-embedding-3-large`

LLM
- Provider selezionabile: **OpenAI** oppure **Ollama (locale)**.
- Selettore del modello LLM e **slider “Temperature”** in sidebar.
- Gestione robusta delle risposte Ollama (`stream=False` + fallback per JSON concatenati) per evitare errori tipo “Extra data”.

Chat RAG
- Top‑k configurabile e **soglia massima di distanza** per filtrare i vicini: se nessun risultato è sotto soglia, la lista “Risultati simili” non viene mostrata.
- **Toggle “Mostra prompt LLM”** in sidebar: permette di visualizzare, in un expander, il prompt effettivamente inviato al modello.
- **Fix duplicazione output**: una sola generazione risposta e un’unica lista di risultati simili (cliccabili) quando presenti.

API Keys
- **Override OpenAI API Key dalla sidebar** (textbox “OpenAI API Key”): prioritaria rispetto alle variabili d’ambiente; disabilitata automaticamente quando non necessaria (LLM=Ollama **e** Embeddings=Locali).

Utility e UX
- Pulsante **Quit** in sidebar.
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
## 🧰 Requisiti e installazione

## 🔹 ️ Dipendenze
- `streamlit`, `requests`, `chromadb`, `sentence-transformers`, `openai`, `pandas` (Ollama opzionale per LLM locali).

Installazione
```bash
pip install -U streamlit requests chromadb sentence-transformers openai tiktoken pandas
# opzionale, per usare LLM locali:
# installa ed esegui Ollama sul sistema
```

## 🔹 ▶️ Avvio
```bash
streamlit run app.py --server.port 8502
```

Modalità CLI (fallback)
- Senza Streamlit, il file espone self‑tests minimi eseguibili da terminale.

---

## ⚡️ Configurazione rapida

1. Inserisci in sidebar l’URL e il Token (Bearer) di YouTrack.
2. Seleziona/crea una **collection** nel Vector DB (Chroma) e verifica l’apertura automatica.
3. Scegli **Embeddings** e **LLM** (OpenAI o Ollama). Se necessario, inserisci la **OpenAI API Key** in sidebar (ha priorità sull’ambiente).
4. Carica un progetto: i ticket vengono **auto‑caricati** e mostrati in tabella.
5. Premi **Indicizza ticket** per popolare la collection (solo la prima volta o per aggiornare).
6. Vai alla sezione **Chatbot RAG**: inserisci il testo del nuovo ticket e premi **Cerca e rispondi**.
   - Opzionale: abilita **Mostra prompt LLM** per vedere il prompt finale.
   - Regola la **soglia distanza** per filtrare i risultati simili mostrati.

---

## 🔒 Sicurezza

Gestione credenziali
- **OpenAI API Key**: può essere fornita da variabile d’ambiente o inserita in **sidebar**; la chiave inserita in UI **non viene salvata su disco** e resta nello `st.session_state`.
- **YouTrack Token**: inserito in sidebar, usato solo per le chiamate API e **non** scritto su file.

Dati e log
- I ticket vengono indicizzati **localmente** in Chroma (path configurabile). Evita di condividere il datastore se contiene dati sensibili.
- La visualizzazione del prompt è **opt‑in** (disattivata di default). Evita di includere dati sensibili nel prompt.

LLM locali vs cloud
- Con **Ollama** i dati **non** lasciano la macchina.
- Con **OpenAI**, prompt e contesti vengono inviati al servizio cloud del provider scelto.

---

## 📈 Stato attuale

Completato
- Connessione a YouTrack, selezione progetto e **auto‑caricamento** ticket con tabella e link.
- Gestione Chroma: selezione/creazione/eliminazione collection, **apertura automatica** senza re‑indicizzazione obbligatoria (fix del naming).
- Indicizzazione con scelta embeddings; **meta provider/modello** e avviso in caso di mismatch durante la query.
- Chat RAG con **soglia distanza**, **visualizzazione prompt** a richiesta e **temperature** per LLM.
- Gestione robusta Ollama (no JSON concatenati).
- **Override OpenAI API Key** dalla sidebar con abilitazione/disabilitazione dinamica.

In corso / raccomandazioni
- **Deduplica/upsert**: preferire `upsert` se la versione di Chroma lo supporta; in alternativa, filtrare gli ID esistenti o fare `delete+add` per aggiornare documenti modificati.
- **Chunking / Re‑rank / MMR**: non attivati di default; da valutare per descrizioni molto lunghe o risultati troppo omogenei.

---

## 🛠️ Troubleshooting

- “Extra data: line … column …” con Ollama → assicurati che l’app usi `stream=False` nell’endpoint `/api/chat` (già integrato) e che l’endpoint risponda con JSON singolo.
- Nessun risultato simile ma appaiono “match” non pertinenti → alza (numericamente) la **soglia distanza** per filtrare i vicini lontani.
- Duplicazione dell’output → risolta nelle versioni recenti (un’unica generazione e una sola lista di risultati).

---

## 🗂️ Struttura progetto

- `app.py` — applicazione Streamlit completa (UI, ingestion, retrieval, chat, CLI).
- `data/chroma/` — persistenza Chroma (inclusi file meta `collection__meta.json`).
- `README.md` — questo documento.

---

## 📄 Licenza

Vedi file LICENSE se presente nel repository.