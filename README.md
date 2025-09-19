# Streamlit RAG Support App

Questa applicazione è una **app standalone basata su Streamlit** che permette di gestire richieste di supporto
sulla base di risposte già fornite in passato. Integra **YouTrack**, un **Vector DB locale** e modelli LLM
(OpenAI e Ollama) per fornire risposte contestualizzate ai nuovi ticket.

---

## 🚀 Funzionalità principali

- **Connessione a YouTrack**: inserendo l’URL del server e un token, è possibile caricare la lista dei progetti e i relativi ticket.
- **Gestione progetti e ticket**: l’utente può selezionare un progetto e visualizzare i ticket associati.
- **Vector Database (Chroma)**:
  - I ticket vengono indicizzati in un database vettoriale persistente (`data/chroma/chroma.sqlite3`).
  - Se il DB è già presente, viene caricato automaticamente e mostrato il numero di documenti presenti.
- **Embeddings**:
  - Possibilità di scegliere tra embeddings locali (`sentence-transformers`) o cloud (`OpenAI Embeddings`).
- **LLM (Large Language Model)**:
  - Selezione tra **OpenAI** (cloud) o **Ollama** (locale).
  - Le risposte citano sempre gli ID dei ticket simili recuperati.
- **Chatbot RAG**:
  - L’utente può inserire un nuovo ticket.
  - Il sistema recupera i ticket più simili e genera una risposta utilizzando l’LLM selezionato.
- **Interfaccia Streamlit**:
  - Sidebar con configurazioni e pulsante **Quit** per chiudere l’app direttamente dal browser.
  - Tabelle adattive (`st.dataframe(width=...)`) per una migliore visualizzazione.
- **Modalità CLI**:
  - Se Streamlit non è disponibile, l’app entra in modalità CLI con self-test delle principali funzioni.

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

## ⚙️ Dipendenze

Per installare tutte le dipendenze necessarie:

```bash
pip install -U streamlit requests chromadb sentence-transformers openai tiktoken pandas
# opzionale per modelli locali
pip install -U ollama
```

---

## ▶️ Avvio

Eseguire in modalità interfaccia Streamlit:

```bash
streamlit run app.py --server.port 8502
```

Oppure avviare in modalità CLI (senza UI Streamlit):

```bash
python app.py
```

---

## 📂 Struttura dati

- `data/chroma/chroma.sqlite3` → database persistente dei ticket indicizzati.
- `app.py` → codice principale dell’applicazione.
- `README.md` → questo file di descrizione.

---

## 🔒 Sicurezza

- La chiave **OPENAI_API_KEY** deve essere impostata come variabile di ambiente.
- Non inserire la chiave hardcoded nel codice.

---

## ✅ Stato attuale

- Connessione a YouTrack funzionante
- Indicizzazione dei ticket in Chroma
- Recupero automatico del DB se già presente
- Generazione embeddings (OpenAI / locali)
- LLM attivi (OpenAI GPT e Ollama)
- Chatbot RAG operativo
