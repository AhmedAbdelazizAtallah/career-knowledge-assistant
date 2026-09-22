# Career Knowledge Assistant

A RAG (Retrieval-Augmented Generation) chat app that answers career questions
(resumes, interviews, salary negotiation, career growth) grounded strictly in
the PDF guides in `data/`. Runs entirely on Cohere's API (embeddings, rerank,
chat via `ClientV2`) with a persistent Chroma + BM25 hybrid retriever, so
it's light enough for a free hosting tier with no GPU and no external
database service.

## Architecture

```
PDF (data/*.pdf)
 -> core/ingestion.py     parse (pypdf + pdfplumber tables) -> strip repeated
                           headers/footers -> clean -> structure-aware,
                           parent-child chunking -> metadata
 -> core/embeddings.py    Cohere embeddings (configurable model)
 -> core/vectorstore.py   persistent Chroma vector store (chroma_db/)

User query
 -> core/query_processing.py   conversational query rewriting
 -> core/retriever.py          hybrid retrieval: Chroma dense + BM25 sparse,
                                 fused with Reciprocal Rank Fusion
 -> core/reranker.py           Cohere Rerank narrows the shortlist
 -> (parent-child expansion: a relevant chunk's full section is sent, not
     just the fragment that matched)
 -> core/generator.py          grounded Cohere chat + citations
 -> core/observability.py      full per-query trace (logs/traces.jsonl)

core/pipeline.py     wires all of the above into one answer_query() call
app.py               Streamlit UI -- calls core.* only, never touches secrets
scripts/ingest.py    one-time/on-demand CLI to (re-)embed data/ into Chroma
eval/                small labeled eval set + retrieval/answer-quality harness
data/                 Source PDFs (the knowledge base)
Cohere_RAG.ipynb      Notebook walkthrough of the same pipeline, cell by cell
```

**Chunking** (`core/ingestion.py:build_chunks()`) splits at real section
boundaries (numbered headers like "2. Format & Structure") before falling
back to size-based splitting within an over-long section. A section header
is **carried across page breaks**, so a section that continues onto a new
page with no header of its own still gets the right label and heading
context instead of silently losing both. Every child chunk also links to
its full section (`parent_id`/`parent_text`) for **parent-child
retrieval**: once any fragment of a section is judged relevant, the
generator sees the whole section, not an arbitrary page-sized slice of it.
Tables are detected separately via `pdfplumber` and rendered as Markdown so
they survive parsing intact instead of being flattened into ambiguous
prose. Repeated running headers/footers and bare page-number lines are
stripped before chunking so they don't pollute every chunk's embedding.

**Query rewriting** (`core/query_processing.py:rewrite_query()`) turns a
follow-up like *"what about the second one?"* into a standalone query using
only the prior conversation turns -- never inventing new facts, and
returning the question unchanged when it's already standalone or when the
rewrite call itself fails.

**Retrieval pipeline** (`core/retriever.py:retrieve()`): the (rewritten)
query goes through a hybrid stage -- Chroma dense vector search and BM25
sparse keyword search, fused with Reciprocal Rank Fusion, filtered by a
relevance gate so an off-topic query can get **zero** candidates. Whatever
survives is passed through **Cohere Rerank** (`rerank-v3.5`), which narrows
a noisy top-20 shortlist down to the ~5 chunks that are genuinely most
relevant, then deduplicated and expanded to their full parent section.

**Citations** (`core/generator.py:generate_answer()`) use Cohere's native
`documents=`/`citations=` grounding: the model can only cite document IDs
we actually supplied, so a hallucinated filename is structurally
impossible. Citations appear two ways: inline markers after each paragraph
(`[Source: doc_title, Section: heading]`) using the citation offsets Cohere
returns, and a numbered source list (`[1] document.pdf — Page 12`) built
from the same verified citations.

**Persistence**: embeddings are stored in a local, disk-persisted Chroma
collection (`chroma_db/`, gitignored) via `core/vectorstore.py`. The app
only *reads* that store at startup (`core/retriever.py:build_index()`) --
run `python scripts/ingest.py` whenever `data/` changes, rather than
re-embedding on every process start. If the store happens to be empty
(e.g. a fresh clone on Streamlit Cloud, whose filesystem resets on
redeploy), `build_index()` ingests automatically on first load.

**Observability** (`core/observability.py`): every call to
`core/pipeline.py:answer_query()` appends one JSON line to
`logs/traces.jsonl` with the original query, rewritten query, every
retrieved chunk's scores (vector/BM25/RRF/rerank), the final context chunk
ids, the answer, citations, and per-stage latency -- enough to debug a bad
answer after the fact without reproducing it live.

**Security** (see also "Security model" below): retrieved PDF text is
passed to the model via Cohere's `documents=` parameter, a channel
structurally separate from `messages=`; the system prompt explicitly
instructs the model to treat that content as data, never as instructions,
so a prompt-injection payload hidden in a PDF cannot override the system
prompt. Malformed PDFs, unreachable vector/BM25 stores, and Cohere API
failures are all caught and degrade to a clear message rather than a raw
crash (see `core/ingestion.py`, `core/vectorstore.py`, `core/llm_utils.py`).

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

Build the vector store once (re-run whenever `data/` changes; `--rebuild`
wipes and re-embeds from scratch):

```bash
python scripts/ingest.py
```

Run it:

```bash
streamlit run app.py
```

## Configuration

All of the following are optional env vars (set in `.env` or your
deployment platform's secrets manager) with sensible defaults baked in:

| Variable | Default | Controls |
|---|---|---|
| `COHERE_EMBED_MODEL` | `embed-english-v3.0` | Embedding model (`core/embeddings.py`) |
| `COHERE_RERANK_MODEL` | `rerank-v3.5` | Reranker model (`core/reranker.py`) |
| `COHERE_CHAT_MODEL` | `command-r-08-2024` | Chat + query-rewrite model |
| `RETRIEVAL_CANDIDATE_POOL` | `20` | Hybrid-stage shortlist size before reranking |
| `RETRIEVAL_TOP_K` | `5` | Final chunks (post-rerank, post-parent-dedup) sent to the LLM |
| `VECTOR_SIM_THRESHOLD` | `30.0` | Min dense similarity (%) to enter the shortlist |
| `BM25_SCORE_THRESHOLD` | `1.0` | Min BM25 score to enter the shortlist |
| `RERANK_SCORE_THRESHOLD` | `0.15` | Min Cohere rerank relevance to reach the LLM |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Where the vector store is written on disk |
| `CHROMA_COLLECTION` | `career_knowledge_base` | Chroma collection name |
| `RAG_TRACE_LOG` | `logs/traces.jsonl` | Observability trace log path |

## Evaluation

`eval/dataset.json` has 16 hand-written, source-grounded questions spanning
all 7 PDFs (plus one deliberately out-of-corpus question, to check that the
pipeline correctly refuses instead of guessing). `eval/run_eval.py` always
builds an **isolated, disposable** vector store for the run -- it never
touches the app's persisted `chroma_db/` -- so it's safe to sweep
parameters freely:

```bash
# Full run: retrieval metrics + LLM-judged answer quality
python eval/run_eval.py

# Fast retrieval-only sweep (no generation/judge calls)
python eval/run_eval.py --skip-answer-metrics --chunk-size 400 --chunk-overlap 80 --tag small-chunks
python eval/run_eval.py --skip-answer-metrics --mode vector --tag vector-only
python eval/run_eval.py --skip-answer-metrics --mode bm25 --tag bm25-only
python eval/run_eval.py --skip-answer-metrics --no-rerank --tag no-rerank
python eval/run_eval.py --skip-answer-metrics --top-k 3 --tag top3

# On a trial API key (10 calls/min), pace a full run to avoid 429s:
python eval/run_eval.py --delay-seconds 8
```

Each run writes `eval/results/<tag>.json` (per-question rows + a summary)
so different configs can be diffed. Metrics reported:

- **Recall@K / MRR / nDCG@K** -- computed from what `retrieve()` actually
  returns (post-rerank, post-parent-dedup), against each question's labeled
  `relevant_document`/`relevant_pages`.
- **answer_correctness / faithfulness** -- LLM-judged (0-1) via the same
  Cohere client: does the answer convey the expected facts, and is every
  claim in it supported by the retrieved context.
- **citation_correctness** -- deterministic check that every cited document
  actually appears among the chunks that were retrieved for that query.
- **refusal_accuracy** -- whether the out-of-corpus question was correctly
  declined rather than answered.

On this dataset, the default config (hybrid + rerank, top-5) scores
Recall@5 = 1.0, MRR = 0.97, nDCG@5 = 0.94, answer_correctness = 0.81,
faithfulness = 0.92, citation_correctness = 1.0, and refusal_accuracy = 1.0
(see `eval/results/` after running).

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

Note: `chroma_db/` is gitignored, so a fresh deploy starts with an empty
vector store. `core/retriever.py:build_index()` detects that and ingests
automatically on first load (a few seconds for this corpus) -- the store
then persists on disk for the life of that container, so subsequent
reruns/sessions don't re-embed. A full redeploy resets the container's
filesystem, so it re-ingests once again on the next cold start.

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
- BM25 is rebuilt in memory from the persisted vector store's contents at
  every process start (cheap -- pure tokenization, no API calls) rather
  than persisted itself; at a much larger corpus size this rebuild would
  need to move to a background job instead of blocking app startup.
- `chroma_db/` is gitignored and lives on the container's local disk, so it
  survives process restarts but not a full redeploy (Streamlit Cloud resets
  the filesystem on redeploy) -- the app just re-ingests once automatically
  on the next cold start. For guaranteed persistence across redeploys
  without re-ingesting, point `CHROMA_PERSIST_DIR`/`CHROMA_COLLECTION` at a
  hosted Chroma/Qdrant instance instead of local disk.
- A trial Cohere API key is capped at 10 calls/minute, which a full
  `eval/run_eval.py` run (multiple calls per question) can exceed --
  `core/llm_utils.py:call_with_retry()` backs off automatically on a 429,
  or use `--delay-seconds` to pace requests.
- No authentication/rate-limiting on the public app itself — anyone with the
  URL can query it (and consume your Cohere quota, including Rerank calls
  on every query). Add `st.secrets`-gated basic auth or a platform-level
  access control if that matters for your use case.
