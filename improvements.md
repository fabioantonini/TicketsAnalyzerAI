Must have (per una PoC convincente)

1. Sync incrementale e upsert

* Scarico oltre 500 issue con paginazione (\$skip/\$top) e solo i cambiamenti recenti (updatedSince). Eviti doppioni usando upsert (o add con try/except su id già presenti) e salvi un last\_sync.json nel persist\_dir.
* Salva nel metadata della collection il nome del modello di embedding e verifica coerenza in query (se cambia modello, avvisa o crea una nuova collection).
* Nota: oggi indicizzi summary+description già correttamente tramite text\_blob, quindi i testi completi entrano nel Vector DB.&#x20;

2. Chunking dei ticket lunghi

* Spezza descrizioni molto grandi in chunk con overlap (per es. 512–800 token con 80 di overlap) e metadati parent\_id, chunk\_id, pos. Migliora recall e riduce “rumore” da descrizioni enciclopediche.

3. Retrieval migliore: MMR e mini re-ranking

* Dopo la query top-k di Chroma, applica MMR (Maximal Marginal Relevance) lato client per diversificare i risultati simili. Facoltativo: usa un CrossEncoder leggero (es. “cross-encoder/ms-marco-MiniLM-L-6-v2”) per ri-ordinare i top-20.

4. UX: link, filtro e feedback

* Rendi cliccabili gli ID dei ticket verso YouTrack (colonna con URL tipo BASE\_URL/youtrack/issue/ID o BASE\_URL/issue/ID, a seconda dell’istanza) e aggiungi un filtro testuale sui ticket caricati.
* Aggiungi thumbs up/down sulla risposta e salva feedback in CSV, così puoi migliorare prompt e valutare top-k hit rate.

5. Robustezza OpenAI e segreti

* Leggi la chiave da st.secrets o da OPENAI\_API\_KEY, mantenendo compatibilità col tuo env var, ma con fallback allo standard; oggi usi OPENAI\_API\_KEY\_EXPERIMENTS sia per embeddings che LLM.&#x20;
* Il tuo LLMBackend già tenta prima Responses API e poi fa fallback su chat.completions; mantieni, ma rendi l’estrazione del testo più resiliente.&#x20;
* Non salvare mai il token YouTrack nei metadati del Vector DB.

6. Cache e prestazioni

* st.cache\_resource per client Chroma, SentenceTransformer e YouTrackClient; st.cache\_data per list\_projects/list\_issues. Migliora molto la reattività su ricarichi.

Nice to have
7\) Hybrid search

* Semplice BM25/keywords pre-filter sui candidate (per esempio con rapidfuzz o rank-bm25 in memoria), poi rerank semantico. Utile quando i testi sono tecnici con termini “forti”.

8. Modelli multilingua e configurabilità

* Default embedding più adatto all’italiano (paraphrase-multilingual-MiniLM-L12-v2) selezionabile da UI; mantiene compatibilità col tuo default all-MiniLM-L6-v2.

9. Prompting e sicurezza

* Prompt injection guard: racchiudi i passaggi di contesto tra tag “Context Begin/End” e chiedi sempre citazioni \[ID-TICKET] (già previsto dal system prompt). Taglia output e invita l’utente a precisare quando il contesto è debole.

10. Valutazioni rapide integrate

* Aggiungi una tab “Eval” con 5–10 query dorate (domande frequenti reali) e mostra precision\@k, MRR e nDCG. Un grafico e una tabella bastano a vendere la PoC.

11. UI polishing

* Loader e progress bar durante ingest; badge in sidebar con count documenti (già mostri il count se esiste la sqlite, ottimo).
* Dataframe largo e con colonne configurate; hai già layout="wide" e passi width alla dataframe, bene.&#x20;

Patch minime subito utili (solo frammenti da incollare)

1. Segreti e chiavi OpenAI più robusti

* Sostituisci la lettura della chiave dove oggi usi OPENAI\_API\_KEY\_EXPERIMENTS con questa utility e chiamala sia in EmbeddingBackend sia in LLMBackend.

  def \_get\_openai\_key():
  val = os.getenv("OPENAI\_API\_KEY") or os.getenv("OPENAI\_API\_KEY\_EXPERIMENTS")
  try:
  import streamlit as st  # se disponibile
  val = st.secrets.get("OPENAI\_API\_KEY", val)
  except Exception:
  pass
  if not val:
  raise RuntimeError("OPENAI\_API\_KEY non impostata")
  return val

2. Paginate e sync incrementale YouTrack

* Aggiungi questi parametri a list\_issues e un ciclo:

  def list\_issues(self, project\_key: str, limit: int = 200, updated\_since: Optional\[str] = None) -> List\[YTIssue]:
  fields = "idReadable,summary,description,updated,project(name,shortName)"
  base\_params = { "query": f"project: {project\_key}", "\$top": str(limit), "fields": fields }
  if updated\_since:
  base\_params\["query"] += f" updated: {updated\_since} .. now()"
  issues = \[]
  skip = 0
  while True:
  params = dict(base\_params, \*\*{"\$skip": str(skip)})
  data = self.\_get("/api/issues", params=params)
  if not data:
  break
  for it in data:
  issues.append(YTIssue(
  id\_readable=it.get("idReadable",""),
  summary=it.get("summary",""),
  description=it.get("description","") or "",
  project=(it.get("project",{}) or {}).get("shortName") or (it.get("project",{}) or {}).get("name",""),
  ))
  if len(data) < limit:
  break
  skip += limit
  return issues


3. Chunking durante ingest

* Prima di calcolare gli embedding:

  def \_chunk(text, size=1200, overlap=200):
  chunks = \[]
  start = 0
  n = len(text)
  while start < n:
  end = min(start + size, n)
  chunks.append(text\[start\:end])
  start = end - overlap
  if start < 0: start = 0
  return chunks

  ids, docs, metas = \[], \[], \[]
  for it in st.session\_state\["issues"]:
  blob = it.text\_blob()
  for i, ch in enumerate(\_chunk(blob)):
  ids.append(f"{it.id\_readable}::chunk{i}")
  docs.append(ch)
  metas.append({"id\_readable": it.id\_readable, "summary": it.summary, "project": it.project, "chunk": i})

4. MMR dopo la query

* Dopo aver ottenuto docs/metas/dists, applica:

  import numpy as np

  def mmr(query\_vec, doc\_vecs, lambda\_mult=0.5, k=5):
  selected, candidates = \[], list(range(len(doc\_vecs)))
  sims = np.array(\[float(np.dot(query\_vec, v) / (np.linalg.norm(query\_vec)\*np.linalg.norm(v) + 1e-8)) for v in doc\_vecs])
  while candidates and len(selected) < k:
  if not selected:
  i = int(np.argmax(sims\[candidates]))
  selected.append(candidates.pop(i))
  continue
  max\_div = \[]
  for c in candidates:
  div = max(\[float(np.dot(doc\_vecs\[c], doc\_vecs\[s]) / (np.linalg.norm(doc\_vecs\[c])\*np.linalg.norm(doc\_vecs\[s]) + 1e-8)) for s in selected]) if selected else 0.0
  score = lambda\_mult \* sims\[c] - (1 - lambda\_mult) \* div
  max\_div.append(score)
  idx = int(np.argmax(max\_div))
  selected.append(candidates.pop(idx))
  return selected

* Recupera anche gli embedding dei vicini da Chroma (o ricalcola lato client sugli stessi documenti) e riordina retrieved secondo mmr(...).

5. Link agli issue nella tabella

* Quando costruisci il DataFrame, aggiungi una colonna url calcolata da yt\_url e id\_readable, poi mostra le righe come markdown cliccabile nella colonna id (oppure usa st.data\_editor con column\_config.LinkColumn).&#x20;

6. Cache veloce

* Decora list\_projects e list\_issues con st.cache\_data e costruzione di Chroma/SentenceTransformer con st.cache\_resource per evitare ricostruzioni ad ogni rerun.

7. UI qualità della vita

* Aggiungi un text\_input per filtrare i ticket mostrati (per summary/description) prima dell’ingest.
* Sostituisci os.\_exit(0) con una nota: chiudere dalla console o esporre un endpoint di shutdown; os.\_exit può troncare senza cleanup.&#x20;

Note puntuali sul codice attuale

* L’UI è già wide e la tabella usa un width parametrico; bene per evitare la “finestra stretta”.&#x20;
* L’ingest indicizza summary e description (tramite text\_blob), quindi non solo i titoli.&#x20;
* LLMBackend: ottimo il fallback Responses → chat.completions; rendi solo più robusta l’estrazione del testo e centralizza la gestione della chiave.&#x20;

# **Roadmap Memoria**

# Livello A — Sticky prefs (✅ fatto)

* Preferenze non sensibili su `.app_prefs.json`, reset automatico dei modelli al cambio provider, validazioni.
  *(nulla da fare qui, se non eventuali piccoli tweak UX).*

# Livello B — “Solution memory” (playbook) (🟡 quasi completo)

**Già fatto**

* Collection separata `memories` (Chroma) con TTL.
* Salvataggio playbook (“Segna come risolto”) con condensazione 3–6 frasi.
* Retrieval combinato KB⊕MEM (stessa soglia), provenienza [MEM], anteprima/expander.
* Vista “Playbook salvati” (tabella), delete-all, sticky toggle per mostrare/nascondere.

**Mancanti/consigliati**

1. **Qualità e tagging avanzato**

   * Campo `quality` gestito da UI: `verified|draft` (ora hardcoded).
   * Tagging manuale in salvataggio (textbox “Tag aggiuntivi” → salva CSV).
   * Filtro per `project`/`tags` in retrieval MEM (es. limita a progetto corrente se presente).

2. **Gestione per-singolo playbook**

   * **Cancella singolo**: pulsante cestino accanto a ogni riga nella tabella.
   * **Modifica testo**: expander con `st.text_area` e bottone “Aggiorna” (ri-salva documento e, opzionalmente, re-embed).

3. **Compatibilità embedder**

   * Aggiungi ai metadati: `mem_embed_provider`, `mem_embed_model`.
   * Avviso in query se embedder MEM ≠ embedder corrente (“le distanze potrebbero degradare”).
   * **Re-embed on-demand**: bottone “Ricalcola embedding” per playbook selezionato quando cambi modello.

4. **UX/Performance elenco**

   * **Paginazione** (batch da 200 → “Carica altri”).
   * **Ricerca full-text** locale (filter client-side su `doc`/`tags`/`project`).
   * **Export** playbook (CSV/Markdown) dalla tabella.

5. **Merging e ranking**

   * MMR leggero sul merged KB⊕MEM (evita ridondanze): diversità sulla base di cosine.
   * **Cap configurabile** per MEM (0–3) + soglia dedicata opzionale (oggi uguale a KB).

6. **Conferma salvataggio & policy**

   * Dialog opzionale “Confermi di memorizzare?” con reminder NO-PII.
   * Template di condensazione più “procedurale” (step numerati, prerequisiti, verifiche).

7. **Metriche (debug facoltativo)**

   * Caption con distanze MEM, tempo embedding, tempo query.
   * Contatore utilizzo playbook (incrementa `uses` nel metadata quando selezionato nel merged).

# Livello C — “Facts strutturati” (🔜 da implementare)

**Obiettivo**: memorizzare **coppie chiave–valore** affidabili (facts) separate dai playbook e inserirle nel prompt come **contesto strutturato**.

**Design proposto**

* **Storage**: piccolo SQLite (`facts.db`) con tabella:

  * `facts(user TEXT, project TEXT, key TEXT, value TEXT, source_ticket TEXT, created_at INT, expires_at INT, confidence REAL DEFAULT 1.0, UNIQUE(user, project, key))`.
* **UI**

  * In chat: pillola “➕ Aggiungi fatto” → modale con (project, key, value, TTL).
  * Lista “Fatti attivi per il progetto X” con pulsanti **Modifica** / **Dimentica** (delete) / **Estendi TTL**.
* **Prompting**

  * Prima della generazione, serializza i facts rilevanti in 2–4 righe concise (es. `project=NETKB; gateway_vendor=Acme; vlan_voice=120; ...`).
  * Se presenti facts, **indicali** al modello come constraints (“non contraddire i facts”).
* **Scadenze & guardrail**

  * TTL automatico; filtro su `expires_at`.
  * No PII/secret; avviso se `key` o `value` sembrano credenziali (regex banali).
* **Ricerca**

  * Usa facts per **boost** nel retrieval (es. se `product=XYZ` → favorisci documenti con `tags` corrispondenti).
* **Operazioni**

  * `upsert` per key per-progetto.
  * “Dimentica questo fatto” in un click.

# Piano pragmatico (2 mini-sprint)

**Sprint 1 (Livello B rifiniture)**

* Per-playbook: elimina/modifica/ri-embed; tag “quality” in UI; ricerca/paginazione; export CSV.
* Metadati embedder + avviso mismatch.
* Cap MEM configurabile + MMR basico.

**Sprint 2 (Livello C MVP)**

* SQLite + CRUD facts; UI “Aggiungi fatto”/“Dimentica”.
* Serializzazione facts nel prompt; TTL/scadenze; validazioni basiche.
* (Opzionale) boost nel retrieval basato su facts.

# **PDF in KB**
Aggiungere PDF può essere **molto utile** — a patto di farlo con alcune cautele per non “sporcare” le risposte. In breve: **separa la sorgente**, **chunking buono**, **provenienza chiara**, e **fusione controllata** con i ticket.

## Quando ha senso

* Manuali, runbook, KB esterne, RFC, guide vendor → **sì** (spesso completano i ticket).
* Allegati rumorosi (log grezzi, report generici) → **no** o metti una soglia/qualità bassa.

## Come integrarle senza confusione

1. **Collection separata**
   Metti i PDF in una collection Chroma distinta (es. `docs`) invece di mescolarli ai ticket (`tickets`) e ai playbook (`memories`).
   → In query fai un **merge KB⊕MEM⊕DOCS** con **cap** per i DOCS (es. max 1–2) e soglia dedicata.

2. **Chunking & overlap**

   * Estrarre testo per pagina, poi spezzare in chunk di **~500–1000 token** con **overlap 100–150**.
   * Conserva **page_number** e **section headings** se disponibili (titolo PDF, capitolo).

3. **Metadati ricchi**

   * `source="pdf"`, `title`, `page`, `project` (se pertinente), `product`, `tags`, `quality="verified|vendor|draft"`, `uri` (path o URL).
   * Questo abilita filtri (es. restringi ai PDF del progetto corrente).

4. **Provenienza chiara in UI**
   Nei “Risultati simili” etichetta come **[DOC]** e rendi cliccabile il `uri` o “PDF p.12”.
   Aggiungi un toggle “**Includi PDF**” in sidebar per chi vuole solo ticket.

5. **Ranking/fusione**

   * Mantieni soglie: **DIST_MAX_DOCS** uguale o leggermente più severa dei ticket.
   * Applica **MMR leggero** nel merge per evitare tre snippet quasi identici dallo stesso PDF.
   * **Cap DOCS** (1–2) così i ticket restano protagonisti.

6. **Modello embeddings**

   * Va bene `all-MiniLM-L6-v2` per PoC; se i PDF sono lunghi/tecnici e vedi risultati deboli, prova `text-embedding-3-large` per i **DOCS** (puoi embeddare docs con un modello e i ticket con un altro; tieni traccia in metadata e segnala il mismatch se cambi embedder in query).

7. **Qualità e sicurezza**

   * Evita PDF scannerizzati senza OCR (o attiva OCR).
   * Non indicizzare PDF con PII/segretI se usi embeddings cloud.
   * Aggiungi un campo `quality` e (opzionale) **TTL** anche per DOCS “temporanei”.

## Mini-roadmap (sicura, in 2 passi)

* **Sprint PDF-1 (ingest separato)**

  * Sidebar: upload o cartella **“Documenti PDF”** → estrazione → chunking → embeddings → **collection `docs`**.
  * Metadati: `title,page,uri,project,tags,quality`.
* **Sprint PDF-2 (query & UI)**

  * Retrieval combinato **KB⊕MEM⊕DOCS** (cap DOCS=1–2, MMR on).
  * Toggle “Includi PDF” + etichetta **[DOC]** in output con link alla pagina.
  * (Opzionale) Expander per mostrare l’estratto completo del chunk.

## (Se vuoi gli snippet)

Posso fornirti subito:

* loader/estrattore PDF con `pypdf` e chunker,
* funzione `ingest_pdfs_to_docs(persist_dir, path_glob="*.pdf")`,
* patch di retrieval con terza sorgente `docs` + cap e rendering `[DOC]`.

In sintesi: **sì, aggiungi i PDF**, ma **tenendoli separati** e con **cap/filtri**. Così ottieni più copertura informativa senza degradare la precisione delle risposte.

