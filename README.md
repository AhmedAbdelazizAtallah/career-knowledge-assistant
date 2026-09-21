# Career Knowledge Assistant

A RAG (Retrieval-Augmented Generation) chat app that answers career questions
(resumes, interviews, salary negotiation, career growth) grounded strictly in
the PDF guides in `data/`. Runs on Cohere's API (embeddings + chat) with a
FAISS + BM25 hybrid retriever, so it's light enough for a free hosting tier
with no GPU and no persistent database.

## Architecture

```
core/rag.py     PDF ingestion, chunking, hybrid retrieval, Cohere API calls
app.py          Streamlit UI -- calls core.rag only, never touches secrets
data/           Source PDFs (the knowledge base)
Ollama.ipynb    Original local-only prototype (Ollama + MiniLM), kept for reference
```

The index (FAISS vectors + BM25) is built **in memory at startup** from the
PDFs in `data/` and cached for the process's lifetime — there's no vector
database file to persist, back up, or go stale.

## Security model

- `COHERE_API_KEY` is read **only** server-side, in `core/rag.py:resolve_secret()`,
  which checks Streamlit's secrets manager first, then the OS environment.
- It is never hardcoded, never logged, never included in an HTTP response,
  and never present in any file that gets committed (`.env` and
  `.streamlit/secrets.toml` are both gitignored).
- `app.py` (the UI layer) never touches the raw key at all — it only calls
  `core.rag` functions and displays their results. There is no code path
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
   runtime — `core/rag.py` reads it automatically via `resolve_secret()`,
   no code changes needed.
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
   fabricating an answer — this is the relevance gate in `core/rag.py:retrieve()`
   working as intended.
4. Check the sidebar shows a nonzero **"Indexed chunks"** count, confirming
   ingestion + embedding ran successfully against your live Cohere key.

## Known limitations

- `VECTOR_SIM_THRESHOLD` (default 30%) was tuned empirically against a
  local MiniLM model in an earlier prototype, not against Cohere's
  `embed-english-v3.0`. After deploying with a real key, run a few
  known-relevant and known-irrelevant queries, check the vector-similarity
  scores logged in `retrieve()`, and adjust `VECTOR_SIM_THRESHOLD` /
  `BM25_SCORE_THRESHOLD` via environment variables if needed (no code
  change required).
- The in-memory index rebuilds (and re-embeds all chunks via the Cohere API)
  on every cold start. For this dataset (~70 chunks) that's a single batched
  API call and takes a couple of seconds — fine at this scale, but would
  need a persisted/cached index for a much larger document set.
- No authentication/rate-limiting on the public app itself — anyone with the
  URL can query it (and consume your Cohere quota). Add
  `st.secrets`-gated basic auth or a platform-level access control if that
  matters for your use case.
