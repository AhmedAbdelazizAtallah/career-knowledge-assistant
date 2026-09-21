"""
Career Knowledge Assistant -- production Streamlit UI over the RAG engine
in core/rag.py (Cohere embeddings + chat, hybrid FAISS/BM25 retrieval).

SECURITY: this file never reads, displays, or forwards COHERE_API_KEY to
the browser. It only calls core.rag functions, which read the key
server-side (see core/rag.py:resolve_secret). No secret ever appears in
this UI layer's code or in anything rendered to the client.

Local run:
    1. cp .env.example .env   and fill in COHERE_API_KEY
    2. pip install -r requirements.txt
    3. streamlit run app.py

Deployment: see README.md for Streamlit Community Cloud / Hugging Face
Spaces steps, including where to paste the key into each platform's
Secrets manager (never into source control).
"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # no-op in deployment (secrets come from the platform instead)

from core.rag import build_index, generate_answer, get_cohere_client, CHAT_MODEL, EMBED_MODEL

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
    "powered by Cohere, deployable on the free tier."
)


@st.cache_resource(show_spinner="Loading knowledge base (embedding documents via Cohere)...")
def load_index():
    client = get_cohere_client()  # raises a clear error if COHERE_API_KEY is missing
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
    st.caption(f"Embed: `{EMBED_MODEL}` · Chat: `{CHAT_MODEL}`")
    st.caption("Hybrid retrieval: FAISS (vector) + BM25 (keyword)")
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
            result = generate_answer(index, prompt, history)
            st.markdown(result["answer"])

            if result["sources"] and not result["grounded"]:
                st.warning(
                    "⚠️ The model answered without citing a specific source from the "
                    "retrieved documents — treat this answer with extra caution."
                )

            if result["cited_sources"]:
                with st.expander(f"📎 {len(result['cited_sources'])} source(s) cited"):
                    for s in result["cited_sources"]:
                        st.caption(f"**{s['file_name']}** — page {s['page_number']}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
