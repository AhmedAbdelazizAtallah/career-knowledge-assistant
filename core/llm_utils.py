"""
core/llm_utils.py -- small helpers shared across every module that calls
the Cohere API (core/generator.py, core/query_processing.py,
core/reranker.py, core/embeddings.py, eval/run_eval.py), so
response-parsing and retry behavior stay identical everywhere instead of
drifting between separate copies.
"""
import time

import cohere

API_RETRY_ATTEMPTS = 2
API_RETRY_BACKOFF_SECONDS = 1.0
# A trial API key is capped at 10 calls/minute -- a short backoff won't
# clear that within the same minute, so a 429 gets a much longer wait
# than an ordinary transient failure.
RATE_LIMIT_BACKOFF_SECONDS = 20.0


def call_with_retry(fn, attempts: int = API_RETRY_ATTEMPTS, backoff_seconds: float = API_RETRY_BACKOFF_SECONDS):
    """Transient network blips and rate-limit hiccups shouldn't surface a
    raw error to the caller on the first failure -- retry a couple of
    times before giving up and letting the caller handle (and clearly
    report) a genuine failure."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                wait = RATE_LIMIT_BACKOFF_SECONDS if isinstance(exc, cohere.errors.TooManyRequestsError) else backoff_seconds * (attempt + 1)
                time.sleep(wait)
    raise last_exc


def extract_text(message) -> str:
    """message.content is either a plain str or a list of content blocks
    (Cohere ClientV2's richer format) -- normalize to plain text either
    way."""
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "".join(getattr(block, "text", "") for block in content)
