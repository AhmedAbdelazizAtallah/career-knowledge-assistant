# Career Knowledge Assistant

A RAG (Retrieval-Augmented Generation) chat app that answers career questions
(resumes, interviews, salary negotiation, career growth) grounded strictly in
the PDF guides in `data/`. Runs entirely on Cohere's API (embeddings, rerank,
chat via `ClientV2`) with a FAISS + BM25 hybrid retriever, so it's light
enough for a free hosting tier with no GPU and no persistent database.

## Architecture

```
core/ingestion.py   PDF parsing, structure-aware chunking, title/section metadata
core/retriever.py   Hybrid search (FAISS + BM25 via RRF) -> Cohere Rerank
core/generator.py   Secret handling, Cohere ClientV2 chat, in-text citations
app.py              Streamlit UI -- calls core.* only, never touches secrets
data/                Source PDFs (the knowledge base)
Cohere_RAG.ipynb     Notebook walkthrough of the same pipeline, cell by cell
```

**Retrieval pipeline** (`core/retriever.py:retrieve()`): a query first goes
through a hybrid stage -- FAISS dense vector search and BM25 sparse keyword
search, fused with Reciprocal Rank Fusion, filtered by a relevance gate so
an off-topic query can get **zero** candidates. Whatever survives that gate
is then passed through **Cohere Rerank** (`rerank-v3.5`), which narrows a
noisy 8-12-chunk shortlist down to the 3-4 chunks that are genuinely most
relevant -- this is what keeps context clean and avoids "lost in the
middle" degradation, rather than just handing the model everything the
hybrid stage found.

**Chunking** (`core/ingestion.py:build_chunks()`) splits at real section
boundaries (numbered headers like "2. Format & Structure") before falling
back to size-based splitting within an over-long section, and every chunk
carries its document title, section heading, and page number as metadata
-- not just a page number.

**Citations** (`core/generator.py:generate_answer()`) use Cohere's native
`documents=`/`citations=` grounding: the model can only cite document IDs
we actually supplied, so a hallucinated filename is structurally
impossible. Citation markers are inserted **inline** in the answer text
(`[Source: doc_title, Section: heading]`) using the citation offsets Cohere
returns, not regex-parsed from free text.

The index (FAISS vectors + BM25) is built **in memory at startup** from the
PDFs in `data/` and cached for the process's lifetime — there's no vector
database file to persist, back up, or go stale.

## Security model

- `COHERE_API_KEY` is read **only** server-side, in
  `core/generator.py:resolve_secret()`, which checks Streamlit's secrets
  manager first, then the OS environment.
- It is never hardcoded, never logged, never included in an HTTP response,
  and never present in any file that gets committed (`.env` and
  `.streamlit/secrets.toml` are both gitignored, along with `*.pyc` /
  `__pycache__/`).
- `app.py` (the UI layer) never touches the raw key at all — it only calls
  `core.*` functions and displays their results. There is no code path
  from the browser back to the key.
- If the key is missing, the app fails with a clear message (see
  screenshot-equivalent below) instead of crashing with a raw traceback or
  silently running with no auth.

## Local setup

```bash
git clone <your-repo-url>
cd NRAG
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your real key (get a free trial key at
[dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)):

```
COHERE_API_KEY=your_real_key_here
```

Run it:

```bash
streamlit run app.py
```

## Deploying for free (Streamlit Community Cloud)

1. **Push to GitHub without secrets.** `.env` is already in `.gitignore`, so
   a normal `git add . && git commit && git push` will never include your
   key. Double-check before pushing:
   ```bash
   git status --short   # .env should NOT appear here
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub.
3. Click **"New app"** → select your repository, branch (`main`), and set
   **Main file path** to `app.py`.
4. Before (or after) deploying, click **"Advanced settings..."** (or, once
   the app exists, open it → **⋮ menu → Settings → Secrets**).
5. In the **Secrets** text box, paste (TOML format):
   ```toml
   COHERE_API_KEY = "your_real_key_here"
   ```
6. Click **Save**. Streamlit Cloud injects this into `st.secrets` at
   runtime — `core/generator.py` reads it automatically via
   `resolve_secret()`, no code changes needed.
7. Click **Deploy**. Your app will be live at
   `https://<your-app-name>.streamlit.app`.

## Deploying for free (Hugging Face Spaces -- alternative)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space), pick
   the **Streamlit** SDK, and create the Space (public or private).
2. Push this repo's contents to the Space's git remote (HF gives you the
   remote URL on the Space page) — same rule: `.env` is gitignored, so your
   key never gets pushed.
3. On the Space page, go to **Settings → Variables and secrets → New
   secret**.
4. Set **Name** to `COHERE_API_KEY` and **Value** to your real key. Click
   **Save**.
5. The Space rebuilds automatically and reads the secret the same way —
   `os.getenv("COHERE_API_KEY")` sees it as a normal environment variable
   inside the container.

## Verification steps

**Confirm the key is never exposed to the client:**
1. Open the deployed app's public URL in a private/incognito window.
2. Open browser DevTools → Network tab, ask the app a question, and inspect
   every request/response. You will only see Streamlit's own websocket
   traffic (the UI framework) — the Cohere API calls happen entirely on the
   server, so the key and the Cohere requests never appear in the browser at
   all.
3. View source / inspect the page: search for "cohere" or your key prefix —
   it won't be found anywhere in client-delivered HTML/JS.
4. In your deployment platform's dashboard, secrets are masked/hidden after
   saving (Streamlit Cloud shows `••••••` once set) — you cannot view the
   raw value again there either, only overwrite it.

**Confirm the app works publicly:**
1. From a different network (e.g. your phone on mobile data, not the same
   Wi-Fi), open the public URL.
2. Ask a real question (e.g. *"What's a good salary negotiation tactic?"*)
   and confirm you get a grounded answer with a "sources cited" expander.
3. Ask an unrelated question (e.g. *"What's the capital of France?"*) and
   confirm it responds with "No relevant information was found" rather than
   fabricating an answer — this is the relevance gate in
   `core/retriever.py:_hybrid_shortlist()` working as intended. Verified
   live: a "boiling point of water on Mars" query got 0 hybrid candidates,
   so Rerank and Chat were never even called for it.
4. Check the sidebar shows a nonzero **"Indexed chunks"** count, confirming
   ingestion + embedding ran successfully against your live Cohere key.

## Known limitations

- `VECTOR_SIM_THRESHOLD` (30%) and `BM25_SCORE_THRESHOLD` (1.0) gate the
  hybrid *shortlist* stage; `RERANK_SCORE_THRESHOLD` (0.15) gates the final
  reranked results and is the more reliable signal, since Cohere's
  `relevance_score` is a calibrated 0..1 probability rather than raw cosine
  similarity. All three were tuned empirically against real Cohere scores
  on this dataset (see `core/retriever.py` comments) — if you swap in a
  different document set, re-check a few known-relevant/irrelevant queries
  and adjust via the env vars if needed (no code change required).
- Retrieval does not incorporate conversation history -- only the raw
  follow-up question text is embedded/searched/reranked (the chat model
  sees history, the retriever doesn't). A vague follow-up like "give me
  one more tip like that" can occasionally retrieve chunks from a
  different document than the previous turn. Fixing this properly needs a
  query-rewriting/condensation step and is a reasonable next enhancement,
  not currently implemented.
- The in-memory index rebuilds (and re-embeds all chunks via the Cohere API)
  on every cold start. For this dataset (~80 chunks) that's a single batched
  API call and takes a couple of seconds — fine at this scale, but would
  need a persisted/cached index for a much larger document set.
- No authentication/rate-limiting on the public app itself — anyone with the
  URL can query it (and consume your Cohere quota, including Rerank calls
  on every query). Add `st.secrets`-gated basic auth or a platform-level
  access control if that matters for your use case.
