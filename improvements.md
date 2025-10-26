# Improvements

## Must have (per una PoC convincente)

### 1) Sync incrementale e upsert
- Scarica oltre **500 issue** con paginazione (`$skip`/`$top`) e solo i cambiamenti recenti (`updatedSince`).
- Evita doppioni usando **upsert** (o `add` con `try/except` su ID già presenti) e salva un `last_sync.json` nel `persist_dir`.
- Salva nel **metadata** della collection il **nome del modello di embedding** e verifica la coerenza in query (se cambia modello, avvisa o crea una nuova collection).
- _Nota_: oggi indicizzi `summary + description` correttamente tramite `text_blob`, quindi i testi completi entrano nel Vector DB.

### 2) Chunking dei ticket lunghi
- Spezza descrizioni molto grandi in **chunk** con overlap (es. **512–800 token** con **80** di overlap).
- Metadati consigliati: `parent_id`, `chunk_id`, `pos`.
- Migliora **recall** e riduce il “rumore” da descrizioni enciclopediche.

### 3) Retrieval migliore: MMR e mini re-ranking
- Dopo la query **top-k** di Chroma, applica **MMR (Maximal Marginal Relevance)** lato client per diversificare i risultati.
- Facoltativo: usa un **CrossEncoder** leggero (es. `cross-encoder/ms-marco-MiniLM-L-6-v2`) per ri-ordinare i top-20.

### 4) UX: link, filtro e feedback
- Rendi **cliccabili** gli ID dei ticket verso YouTrack (colonna con URL tipo `BASE_URL/youtrack/issue/ID` o `BASE_URL/issue/ID` in base all’istanza).
- Aggiungi un **filtro testuale** sui ticket caricati.
- **Thumbs up/down** sulla risposta e salva feedback in CSV per migliorare il prompt e valutare il **top-k hit rate**.

### 5) Robustezza OpenAI e segreti
- Leggi la chiave da **`st.secrets`** o da `OPENAI_API_KEY`, mantenendo compatibilità con l’`env` attuale.
- `LLMBackend`: mantieni il **fallback** _Responses API → Chat Completions_, ma rendi l’estrazione del testo più resiliente.
- **Mai** salvare il token YouTrack nei metadati del Vector DB.

### 6) Cache e prestazioni
- Usa `st.cache_resource` per i client **Chroma**, **SentenceTransformer** e **YouTrackClient**.
- Usa `st.cache_data` per **`list_projects`** / **`list_issues`**.
- Migliora sensibilmente la reattività sui ricarichi.

---

## Nice to have

### 7) Hybrid search
- **BM25/keywords pre-filter** sui candidate (es. con `rapidfuzz` o `rank-bm25`) e poi **rerank** semantico.
- Utile su testi tecnici con termini “forti”.

### 8) Modelli multilingua e configurabilità
- Default embedding più adatto all’italiano: `paraphrase-multilingual-MiniLM-L12-v2` (selezionabile da UI).
- Mantieni compatibilità con l’attuale `all-MiniLM-L6-v2`.

### 9) Prompting e sicurezza
- **Prompt injection guard**: racchiudi il contesto tra tag “Context Begin/End” e chiedi sempre citazioni `[ID-TICKET]` (già previsto dal system prompt).
- Trunca output e invita l’utente a precisare quando il contesto è debole.

### 10) Valutazioni rapide integrate
- Tab **“Eval”** con 5–10 query dorate (FAQ reali) e mostra **precision@k**, **MRR**, **nDCG**.
- Un grafico + una tabella bastano a “vendere” la PoC.

### 11) UI polishing
- Loader e progress bar durante ingest; **badge** in sidebar con count documenti.
- Dataframe largo con colonne configurate (hai già `layout="wide"`: bene).

---

## Alcuni suggerimenti

1) **Segreti e chiavi OpenAI** più robusti.  
2) **Paginazione** e **sync incrementale** YouTrack.  
3) **Chunking** durante ingest.  
4) **MMR** dopo la query (recupera anche gli embedding dei vicini o ricalcola lato client; riordina con `mmr(...)`).  
5) **Link** agli issue nella tabella.  
6) **Cache** veloce:
   - Decora `list_projects` / `list_issues` con `st.cache_data`.
   - Costruisci Chroma/SentenceTransformer con `st.cache_resource`.
7) **Qualità della vita (UI)**:
   - `text_input` per filtrare i ticket mostrati (summary/description) _prima_ dell’ingest.
   - Sostituisci `os._exit(0)` con una nota (chiudere da console) o un endpoint di shutdown; `os._exit` può troncare senza cleanup.

---

## Note puntuali sul codice attuale
- L’UI è già **wide** e la tabella usa un **width** parametrico: bene.  
- L’ingest indicizza **summary** e **description** (via `text_blob`): bene.  
- `LLMBackend`: ottimo il fallback **Responses → Chat Completions**; rendi solo più robusta l’estrazione del testo e centralizza la gestione della chiave.

---

## Roadmap Memoria

### Livello A — Sticky prefs (✅ fatto)
- Preferenze non sensibili su `.app_prefs.json`, reset automatico dei modelli al cambio provider, validazioni.  
  _Nulla da fare, al massimo piccoli tweak UX._

### Livello B — “Solution memory” (playbook) (🟡 quasi completo)

**Già fatto**
- Collection separata **`memories`** (Chroma) con **TTL**.
- Salvataggio playbook (“Segna come risolto”) con **condensazione 3–6 frasi**.
- Retrieval combinato **KB ⊕ MEM** (stessa soglia), provenienza **[MEM]**, **anteprima/expander**.
- Vista **“Playbook salvati”** (tabella), **delete-all**, toggle sticky per mostra/nascondi.

**Mancanti/consigliati**
1. **Qualità e tagging avanzato**
   - Campo `quality` da UI: `verified | draft` (ora hardcoded).
   - Tag aggiuntivi in salvataggio (textbox CSV).
   - Filtro per `project`/`tags` nel retrieval MEM (limita al progetto corrente se presente).
2. **Gestione per-singolo playbook**
   - **Cancella singolo** (icona cestino in tabella).
   - **Modifica testo** (expander + `st.text_area`) e **re-embed** opzionale.
3. **Compatibilità embedder**
   - Metadati: `mem_embed_provider`, `mem_embed_model`.
   - Avviso se MEM ≠ embedder corrente; **re-embed on-demand**.
4. **UX/Performance elenco**
   - **Paginazione** (batch 200) + “Carica altri”.
   - **Ricerca** locale (doc/tags/project).
   - **Export** CSV/Markdown.
5. **Merging e ranking**
   - **MMR** sul merge KB⊕MEM per evitare ridondanze.
   - **Cap** MEM configurabile (0–3) + soglia dedicata opzionale.
6. **Conferma & policy**
   - Dialog “Confermi di memorizzare?” (NO-PII reminder).
   - Template di condensazione più “procedurale”.
7. **Metriche (debug)**
   - Caption con distanze MEM, tempo embedding/query.
   - Contatore di utilizzo (`uses`) nel metadata.

### Livello C — “Facts strutturati” (🔜 da implementare)

**Obiettivo**  
Memorizzare **coppie chiave–valore** affidabili (facts) separate dai playbook e inserirle nel prompt come **contesto strutturato**.

**Design proposto**
- **Storage**: SQLite `facts.db`, tabella:
  ```sql
  facts(user TEXT, project TEXT, key TEXT, value TEXT,
        source_ticket TEXT, created_at INT, expires_at INT,
        confidence REAL DEFAULT 1.0,
        UNIQUE(user, project, key))


* **UI**: in chat pillola “➕ Aggiungi fatto” → modale (project, key, value, TTL).
  Lista “Fatti attivi per X” con **Modifica / Dimentica / Estendi TTL**.
* **Prompting**: prima della risposta serializza i facts in **2–4 righe** (`project=NETKB; gateway_vendor=Acme; ...`) e chiedi al modello di **non contraddirli**.
* **Scadenze & guardrail**: TTL, filtro `expires_at`, **no PII/secret** (regex basiche).
* **Ricerca**: usa facts per **boost** nel retrieval (es. `product=XYZ` → favorisci doc con tag corrispondenti).
* **Operazioni**: `upsert` per key per-progetto, pulsante “Dimentica questo fatto”.

### Piano pragmatico (2 mini-sprint)

**Sprint 1 – Rifiniture Livello B**
Elimina/modifica/ri-embed per playbook; `quality` da UI; ricerca/paginazione; export CSV; metadati embedder + avviso mismatch; cap MEM + MMR.

**Sprint 2 – MVP Livello C**
SQLite + CRUD facts; serializzazione nel prompt; TTL/scadenze; validazioni basiche; (opz.) boost retrieval dai facts.

---

## PDF in KB

Aggiungere PDF può essere **molto utile** — a patto di farlo con cautele per non “sporcare” le risposte. In breve: **sorgente separata**, **chunking corretto**, **provenienza chiara**, **fusione controllata** con i ticket.

### Quando ha senso

* Manuali, runbook, KB esterne, RFC, guide vendor → **sì**.
* Allegati rumorosi (log grezzi, report generici) → **no** o **qualità bassa**.

### Come integrarli senza confusione

1. **Collection separata**: `docs` distinta da `tickets`/`memories`. In query fai **KB ⊕ MEM ⊕ DOCS** con **cap DOCS** (1–2) e **soglia dedicata**.
2. **Chunking & overlap**: ~**500–1000 token**, **overlap 100–150**, conserva `page_number` e titoli/heading.
3. **Metadati ricchi**: `source="pdf"`, `title`, `page`, `project`, `product`, `tags`, `quality`, `uri`.
4. **Provenienza chiara in UI**: etichetta **[DOC]** e link “PDF p.12”; toggle “**Includi PDF**” in sidebar.
5. **Ranking/fusione**: soglia DOCS ~ ticket o più severa, **MMR** sul merge, **cap DOCS** 1–2.
6. **Modello embeddings**: ok `all-MiniLM-L6-v2`; per documenti tecnici lunghi valuta `text-embedding-3-large`.
7. **Qualità & sicurezza**: evita scannerizzati senza OCR; no PII/secret; `quality` e (opz.) **TTL**.

**Mini-roadmap**

* **PDF-1 (ingest)**: upload/cartella “Documenti PDF” → estrazione → chunking → embeddings → collection `docs` (metadati `title,page,uri,project,tags,quality`).
* **PDF-2 (query & UI)**: merge **KB⊕MEM⊕DOCS**, cap DOCS=1–2, MMR on; toggle “Includi PDF”; rendering `[DOC]`.

---

## Metriche di valutazione

### 1) Dataset di valutazione (gold set)

* **Raccolta** (50–200 casi): per ogni ticket *Q* definisci **`answer_gold`** e **`doc_gold`**; includi casi facili e rumorosi.
* **Normalizzazione**: pulizia testi solo per matching.
* **Split**: 80% *dev*, 20% *hold-out*.

> Se usi playbook, aggiungi `mem_gold` per ~30% dei casi.

### 2) Valutazione Retrieval (senza LLM)

* Esegui la pipeline di **embedding + nearest neighbors** su `tickets` (e `memories` se attivo).
* Metriche KB: **Recall@k**, **Hit@k**, **MRR@k**, **nDCG@k**.
* Metriche MEM: **Hit@k(mem)**, **Coverage MEM**.
* Analisi soglie: curva **Recall@k** variando **distanza max** e **top-k**; scegli il **punto di lavoro**.
* A/B tecnici: confronta embedder/parametri sullo stesso gold.

### 3) Valutazione Generazione (con LLM)

* **EM/F1** per risposte brevi; **ROUGE-L** per discorsive.
* **Groundedness & Faithfulness**: % frasi supportate, no-contradiction (NLI/regex), **attribution rate**.
* Target: **EM/F1 ≥ 0.6**, **ROUGE-L ≥ 0.35**, **Groundedness ≥ 0.85**, **Contradiction ≈ 0%**.
* **Valutazione umana**: griglia 1–5 (Correttezza/Completezza/Chiarezza/Azione), 2 valutatori su 30–50 casi.

### 4) Valutazione del merge KB ⊕ MEM

* Confronta: (1) Solo KB, (2) KB⊕MEM cap 1, (3) KB⊕MEM cap 2 (+MMR).
* Misura: **F1/ROUGE**, **Groundedness**, **tempo medio**; accetta variante che migliora qualità senza degradare troppo latenza (+10–20%).

### 5) Robustezza & regressioni

* Test avversari (query vaghe/off-topic/typo) → output “no match” puliti.
* Stabilità: ripeti query (stessa temperatura) e verifica varianza.
* Script benchmark che genera **report CSV/HTML** a ogni modifica.

### 6) Telemetria operativa (prod-like)

* Logging (opt-in): query, k, soglia, doc usati, distanza, latenza E2E, provider LLM/embeddings, thumbs up/down.
* Dashboard (anche Streamlit): **hit@k** giornaliero, groundedness stimata, tempi medi, errori.

### 7) Criteri di accettazione (PoC → Go/NoGo)

* Retrieval: **Recall@5 ≥ 0.80**, **MRR@10 ≥ 0.6**.
* Generazione: **Groundedness ≥ 0.85**, **Contradiction ≈ 0%**, **Human score ≥ 4/5**.
* Operatività: **p95 latenza ≤ 4–6 s**, **error rate < 1%**.
* Memoria: **Coverage MEM ≥ 30%**; nessuna memoria contraddittoria.

---

## Mini-howto (subito applicabile)

* **Gold set JSON**:

  ```json
  {
    "query_id": "Q123",
    "query": "VoIP audio monodirezionale con SBC esterno",
    "answer_gold": "Disabilita SIP ALG e apri range RTP.",
    "doc_gold": ["NETKB-3"],
    "mem_gold": []
  }
  ```
* **Script eval (offline)**:

  1. embed → retrieve → Recall/Hit/MRR/nDCG
  2. prompt → LLM → EM/F1/ROUGE
  3. groundedness (overlap/NLI)
  4. CSV con metriche + medie
* **A/B**: flag CLI per embedder, soglia, top-k, cap MEM.

**Consigli tattici**

* **Chunking** ticket lunghi: 200–400 token + overlap 50 → miglior recall.
* **Rerank** top-k (cosine + BM25 o MMR) → migliori MRR/nDCG.
* **Prompt strutturato** (steps, prerequisiti, verifica) → più azionabilità.
* **Soglie diverse per sorgente** (se aggiungi PDF): DOC leggermente più stretta.

---

## Evoluzione UI

### A) Streamlit è “professionale” per un prodotto commerciale?

**Pro**: sviluppo rapido, ottimo per PoC/internal tools, deploy semplice, molti componenti pronti.
**Contro**: multi-utenza/RBAC non nativi, UX avanzata più macchinosa, rerun frequenti, branding limitato.
**Raccomandazione**: Streamlit va benissimo per **team interni/pilot**; per **SaaS/enterprise** valuta stack web classico (React/Next + FastAPI, SSO, worker di ingestion, RAG come servizio).

### B) Migliorare l’UI attuale restando su Streamlit

* **Tabs**: “📥 Ingestion”, “🔎 Ricerca”, “🧠 Playbook”, “⚙️ Config”.
* **Forms** per azioni batch; **expander** per opzioni avanzate; **status bar** con ambiente/provider/collection.
* **Usabilità**: disabilita bottoni senza prerequisiti, toasts chiari, copy-to-clipboard, spinners utili.
* **Stato & performance**: evita scritture a `session_state` dopo i widget; `cache_data` / `cache_resource`; forms per ridurre i rerun.
* **Tabelle**: paginazione/filtri playbook; risultati simili con link cliccabili e expander MEM.
* **Config & sicurezza**: sezioni chiare in sidebar; API key mascherata; esporta/ripristina preferenze.
* **i18n**: valuta switch lingua se servono utenti non italiani.
* **Migrazione product-grade**: segnali—SSO, routing complesso, ingestion async, integrazioni esterne.

**TL;DR**
Rimani su Streamlit per l’attuale target: aggiungi **tabs + forms + status bar**, rafforza cache/stato, rifinisci la tab **Playbook** (paginazione, delete singolo, expander). Prepara un **backend FastAPI** sottile per la logica RAG come step di transizione.
