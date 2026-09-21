"""
core/generator.py -- secure Cohere client creation, prompt engineering,
Cohere Chat completion (ClientV2), and citation handling.

SECURITY: get_cohere_client() is the only place COHERE_API_KEY is read.
It checks Streamlit's secrets manager first, then the OS environment, and
is never logged or returned to a caller -- only an authenticated client
object is exposed. This module has no knowledge of HTTP requests or
browsers, so there is no code path that could leak the key to a client.
"""
import os

import cohere

from core.retriever import retrieve, FINAL_TOP_K

# Cohere periodically retires undated aliases (plain "command-r" was
# removed 2025-09-15) -- if chat starts 404ing with a "model was removed"
# message, check live models with:
#   cohere.ClientV2(api_key=...).models.list(endpoint="chat")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-08-2024")
MAX_HISTORY_TURNS = 3

SYSTEM_PREAMBLE = (
    "You are an expert career consultant. Answer the user's question directly and "
    "comprehensively using ONLY the provided documents, and take the conversation "
    "history into account for follow-up questions.\n"
    "Rules:\n"
    "1. NEVER invent, extrapolate, or fabricate examples, numbers, or facts that are "
    "not present in the provided documents.\n"
    "2. If a formula or framework is given in the documents, present it completely, "
    "including its accompanying rules.\n"
    "3. If the documents do not contain enough information to answer, say so "
    "explicitly instead of guessing."
)


def resolve_secret(name: str) -> str | None:
    """Read a secret from Streamlit's secrets manager if available, else
    from the OS environment. Never accepts a value from a request, a
    client, or a URL -- only server-side sources."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def get_cohere_client() -> cohere.ClientV2:
    api_key = resolve_secret("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. For local runs, copy .env.example to .env and "
            "fill it in (or set the env var directly). For a deployed app, add it in "
            "your platform's secrets manager -- see README.md. It must never be "
            "hardcoded in source or committed to git."
        )
    return cohere.ClientV2(api_key=api_key)


def _build_messages(history: list[dict], user_message: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PREAMBLE}]
    for turn in history[-2 * MAX_HISTORY_TURNS:]:
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def _as_documents(chunks: list[dict]) -> list[dict]:
    """Cohere's native RAG document format. `id` is echoed back in
    citations but never shown to the model itself -- the model can only
    cite ids we handed it, so a hallucinated file name is structurally
    impossible here (unlike asking the model to type a citation as free
    text, which is what the old regex-based approach had to work around)."""
    return [
        {
            "id": c["chunk_id"],
            "data": {
                "text": c["text"],
                "file_name": c["file_name"],
                "document_title": c["document_title"],
                "section": c["section"],
                "page_number": str(c["page_number"]),
            },
        }
        for c in chunks
    ]


def _source_label(chunk: dict) -> str:
    title = chunk.get("document_title") or chunk["file_name"]
    section = chunk.get("section")
    if section:
        return f"[Source: {title}, Section: {section}]"
    return f"[Source: {title}, Page: {chunk['page_number']}]"


def _insert_inline_citations(text: str, citations: list, by_id: dict[str, dict]) -> str:
    """Insert a "[Source: ..., Section: ...]" marker right after each cited
    span, using Cohere's citation offsets. Insertion happens back-to-front
    so earlier offsets aren't invalidated by earlier insertions."""
    inserts: list[tuple[int, str]] = []
    for citation in citations:
        if citation.end is None:
            continue
        labels: list[str] = []
        for source in (citation.sources or []):
            chunk = by_id.get(getattr(source, "id", None))
            if chunk:
                label = _source_label(chunk)
                if label not in labels:
                    labels.append(label)
        if labels:
            inserts.append((citation.end, " " + " ".join(labels)))

    inserts.sort(key=lambda pair: pair[0], reverse=True)
    for pos, label in inserts:
        text = text[:pos] + label + text[pos:]
    return text


def _extract_text(message) -> str:
    """message.content is either a plain str or a list of content blocks
    (V2's richer format) -- normalize to plain text either way."""
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "".join(getattr(block, "text", "") for block in content)


def generate_answer(index, query: str, history: list[dict], top_k: int | None = None) -> dict:
    retrieved = retrieve(index, query, top_k=top_k or FINAL_TOP_K)
    if not retrieved:
        return {
            "answer": "No relevant information was found in the knowledge base for this question.",
            "sources": [], "cited_sources": [], "grounded": False,
        }

    try:
        response = index.client.chat(
            model=CHAT_MODEL,
            messages=_build_messages(history, query),
            documents=_as_documents(retrieved),
            temperature=0.1,
        )
    except Exception as exc:
        return {
            "answer": (
                f"Could not reach Cohere ({CHAT_MODEL}). Check that COHERE_API_KEY is "
                f"valid and has quota remaining. Details: {exc}"
            ),
            "sources": retrieved, "cited_sources": [], "grounded": False,
        }

    by_id = {c["chunk_id"]: c for c in retrieved}
    citations = response.message.citations or []
    raw_text = _extract_text(response.message)
    answer_with_citations = _insert_inline_citations(raw_text, citations, by_id)

    cited_keys: set[tuple[str, str]] = set()
    for citation in citations:
        for source in (citation.sources or []):
            chunk = by_id.get(getattr(source, "id", None))
            if chunk:
                cited_keys.add((chunk["file_name"], chunk["section"] or f"Page {chunk['page_number']}"))

    cited_sources = [{"file_name": f, "location": loc} for f, loc in sorted(cited_keys)]
    return {
        "answer": answer_with_citations,
        "sources": retrieved,
        "cited_sources": cited_sources,
        "grounded": bool(cited_sources),
    }
