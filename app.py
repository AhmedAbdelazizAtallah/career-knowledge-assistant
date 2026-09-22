"""
Career Knowledge Assistant -- production Streamlit UI over the RAG engine
in core/ (structure-aware ingestion -> persistent Chroma vector store +
BM25 hybrid retrieval -> Cohere Rerank -> conversational query rewriting
-> grounded chat with citations). See core/pipeline.py:answer_query for
the full flow and README.md for architecture details.

SECURITY: this file never reads, displays, or forwards COHERE_API_KEY to
the browser. It only calls core.* functions, which read the key
server-side (see core/generator.py:resolve_secret). No secret ever
appears in this UI layer's code or in anything rendered to the client.

Local run:
    1. cp .env.example .env   and fill in COHERE_API_KEY
    2. pip install -r requirements.txt
    3. python scripts/ingest.py   (one-time: embeds data/ into chroma_db/)
    4. streamlit run app.py

Deployment: see README.md for Streamlit Community Cloud / Hugging Face
Spaces steps, including where to paste the key into each platform's
Secrets manager (never into source control).
"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # no-op in deployment (secrets come from the platform instead)

from core.embeddings import EMBED_MODEL
from core.generator import get_cohere_client, CHAT_MODEL
from core.pipeline import answer_query
from core.reranker import RERANK_MODEL
from core.retriever import build_index

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="Career Knowledge Assistant", page_icon="💼", layout="centered")

st.markdown("""
<style>
.stChatMessage { border-radius: 14px; }
h1 { font-weight: 700; letter-spacing: -0.5px; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.2); }
</style>
""", unsafe_allow_html=True)

st.title("💼 Career Knowledge Assistant")
st.caption(
    "RAG chat over resumes, interviews, negotiation & career growth guides — "
    "hybrid search + Cohere Rerank, deployable on the free tier."
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index():
    client = get_cohere_client()  # raises a clear error if COHERE_API_KEY is missing
    # Reads the persistent Chroma store built by scripts/ingest.py; only
    # embeds (and only then) if that store turns out to be empty, e.g. a
    # fresh clone that hasn't been ingested yet.
    return build_index(DATA_DIR, client)


try:
    index = load_index()
except Exception as exc:
    st.error(
        f"Could not start the knowledge base: {exc}\n\n"
        "If this is a missing-key error, see README.md for how to set "
        "`COHERE_API_KEY` locally (.env) or in your deployment platform's secrets manager."
    )
    st.stop()

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.metric("Indexed chunks", len(index))
    st.caption(f"Embed: `{EMBED_MODEL}`")
    st.caption(f"Rerank: `{RERANK_MODEL}`")
    st.caption(f"Chat: `{CHAT_MODEL}`")
    st.caption("Pipeline: rewrite → Chroma + BM25 (RRF) → Cohere Rerank → grounded chat")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about resumes, interviews, salary negotiation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = st.session_state.messages[:-1]
            result = answer_query(index, prompt, history)
            st.markdown(result["answer"])

            if result.get("rewritten_query") and result["rewritten_query"] != prompt:
                st.caption(f"🔎 Searched for: _{result['rewritten_query']}_")

            if result["citations"]:
                st.markdown("**Sources:**  \n" + "  \n".join(result["citations"]))
            elif result["sources"] and not result["grounded"]:
                st.warning(
                    "⚠️ The model answered without citing a specific source from the "
                    "retrieved documents — treat this answer with extra caution."
                )

            if result["sources"]:
                with st.expander(f"📎 {len(result['sources'])} reranked source(s) (debug)"):
                    for s in result["sources"]:
                        page = (
                            f"Page {s['parent_page_start']}"
                            if s["parent_page_start"] == s["parent_page_end"]
                            else f"Pages {s['parent_page_start']}-{s['parent_page_end']}"
                        )
                        st.caption(
                            f"**{s['document_title']}** — {s['section'] or page}  \n"
                            f"relevance: {s.get('rerank_score', 0):.2f} "
                            f"(vector {s['vector_sim']}% · bm25 {s['bm25_score']})"
                        )

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
