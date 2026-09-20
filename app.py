"""
Career Knowledge Assistant -- a simple local chat UI over the RAG pipeline
built in Ollama.ipynb.

Setup:
    1. Run Ollama.ipynb once (top to bottom) to build chroma_db/.
    2. pip install streamlit chromadb sentence-transformers ollama rank_bm25
    3. Make sure `ollama serve` is running and llama3.2:3b is pulled.

Run:
    streamlit run app.py
"""
import re
from pathlib import Path

import chromadb
import ollama
import streamlit as st
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "career_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT_SECONDS = 120
MAX_HISTORY_TURNS = 3

VECTOR_SIM_THRESHOLD = 30.0
BM25_SCORE_THRESHOLD = 1.0
RRF_K = 60

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "at", "by", "from", "into",
    "about", "what", "which", "who", "whom", "how", "when", "where", "why",
    "do", "does", "did", "you", "your", "i", "we", "they", "he", "she", "them",
    "his", "her", "their", "our", "my", "me", "us", "not", "no", "so", "such",
    "than", "too", "very", "can", "will", "would", "should", "could", "may",
    "might", "must", "have", "has", "had",
}

SYSTEM_INSTRUCTION = (
    "You are an expert career consultant. Answer the user's question directly and "
    "comprehensively using ONLY the provided context, and take prior turns in this "
    "conversation into account when relevant (e.g. follow-up questions).\n"
    "Strict Rules:\n"
    "1. NEVER invent, extrapolate, or fabricate any examples. If an example is provided "
    "in the text, quote or adapt ONLY that exact example.\n"
    "2. Present any formula or framework given in the text completely, along with its "
    "accompanying rules.\n"
    "3. Every paragraph or piece of advice MUST end with an explicit source citation in "
    "EXACTLY this format: (exact_file_name.pdf, Page X) -- using the real file name shown "
    "in the 'Source:' header above each context block. For example: "
    "(05_Salary_Negotiation_Playbook.pdf, Page 1). Never write a placeholder like 'Document [1]', "
    "and never add extra words inside the parentheses.\n"
    "4. If the context does not contain enough information to answer, say so explicitly "
    "instead of guessing."
)

_PAREN_PATTERN = re.compile(r"\(([^()]+)\)")
_FILE_PATTERN = re.compile(r"([\w\-]+\.pdf)", re.IGNORECASE)
_PAGE_PATTERN = re.compile(r"Page\s*(\d+)", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


def find_unverifiable_citations(answer: str, retrieved_chunks: list[dict]) -> list[str]:
    valid_pairs = {(c["file_name"].lower(), c["page_number"]) for c in retrieved_chunks}
    problems = []
    for paren_match in _PAREN_PATTERN.finditer(answer):
        content = paren_match.group(1)
        file_match = _FILE_PATTERN.search(content)
        page_match = _PAGE_PATTERN.search(content)
        if not (file_match and page_match):
            continue
        file_name, page = file_match.group(1).lower(), int(page_match.group(1))
        if (file_name, page) not in valid_pairs:
            problems.append(paren_match.group(0))
    if re.search(r"\bDocument\s*\[\d+\]", answer, re.IGNORECASE):
        problems.append("placeholder citation instead of a real file name")
    return problems


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_backend():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    data = collection.get(include=["documents", "metadatas"])
    ids, texts, metas = data["ids"], data["documents"], data["metadatas"]
    token_sets = [set(tokenize(t)) for t in texts]
    bm25_index = BM25Okapi([tokenize(t) for t in texts])
    ollama_client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)

    return collection, ids, texts, metas, token_sets, bm25_index, ollama_client


def retrieve_relevant_chunks(query: str, n_results: int = 4, candidate_pool: int = 10) -> list[dict]:
    collection, all_ids, texts, metas, token_sets, bm25_index, _ = load_backend()
    candidates: dict[str, dict] = {}

    vector_res = collection.query(
        query_texts=[query], n_results=min(candidate_pool, len(texts)),
        include=["documents", "metadatas", "distances"]
    )
    for rank, (cid, doc, meta, dist) in enumerate(zip(
        vector_res["ids"][0], vector_res["documents"][0],
        vector_res["metadatas"][0], vector_res["distances"][0]
    )):
        candidates[cid] = {
            "chunk_id": cid, "text": doc,
            "file_name": meta["file_name"], "page_number": meta["page_number"],
            "vector_sim": round((1 - dist) * 100, 2), "vector_rank": rank,
            "bm25_score": 0.0, "bm25_rank": None,
        }

    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    min_overlap = min(2, len(query_token_set)) if query_token_set else 0

    bm25_scores = bm25_index.get_scores(query_tokens)
    bm25_top = sorted(range(len(texts)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_pool]
    for rank, idx in enumerate(bm25_top):
        if len(query_token_set & token_sets[idx]) < min_overlap:
            continue
        cid = all_ids[idx]
        entry = candidates.setdefault(cid, {
            "chunk_id": cid, "text": texts[idx],
            "file_name": metas[idx]["file_name"], "page_number": metas[idx]["page_number"],
            "vector_sim": 0.0, "vector_rank": None,
            "bm25_score": 0.0, "bm25_rank": None,
        })
        entry["bm25_score"] = round(float(bm25_scores[idx]), 2)
        entry["bm25_rank"] = rank

    def rrf(rank):
        return 0.0 if rank is None else 1.0 / (RRF_K + rank + 1)

    relevant = []
    for c in candidates.values():
        c["score"] = rrf(c["vector_rank"]) + rrf(c["bm25_rank"])
        if c["vector_sim"] >= VECTOR_SIM_THRESHOLD or c["bm25_score"] >= BM25_SCORE_THRESHOLD:
            relevant.append(c)

    relevant.sort(key=lambda c: c["score"], reverse=True)
    return relevant[:n_results]


def build_context(retrieved_chunks: list[dict]) -> str:
    return "\n\n".join(
        f"--- Source: {c['file_name']} (Page {c['page_number']}) ---\n{c['text'].strip()}"
        for c in retrieved_chunks
    )


def generate_rag_response(query: str, history: list[dict], n_results: int = 5) -> dict:
    retrieved_chunks = retrieve_relevant_chunks(query, n_results=n_results)

    if not retrieved_chunks:
        return {
            "answer": "No relevant information was found in the knowledge base for this question.",
            "sources": [], "citation_warnings": [],
        }

    _, _, _, _, _, _, ollama_client = load_backend()
    user_message = f"""Context Documents:
{build_context(retrieved_chunks)}

Question: {query}

Provide a structured, helpful answer based strictly on the context above:"""

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    messages.extend(history[-2 * MAX_HISTORY_TURNS:])
    messages.append({"role": "user", "content": user_message})

    try:
        response = ollama_client.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": 0.1})
    except Exception as exc:
        return {
            "answer": (
                f"Could not reach Ollama model '{OLLAMA_MODEL}'. Make sure `ollama serve` is "
                f"running and the model is pulled (`ollama pull {OLLAMA_MODEL}`). Details: {exc}"
            ),
            "sources": retrieved_chunks, "citation_warnings": [],
        }

    answer = response["message"]["content"]
    return {
        "answer": answer,
        "sources": retrieved_chunks,
        "citation_warnings": find_unverifiable_citations(answer, retrieved_chunks),
    }


# ============================== UI ==============================

st.set_page_config(page_title="Career Knowledge Assistant", page_icon="💼", layout="centered")

st.markdown("""
<style>
.stChatMessage { border-radius: 14px; }
h1 { font-weight: 700; letter-spacing: -0.5px; }
.stCaption { font-size: 0.95rem; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.2); }
</style>
""", unsafe_allow_html=True)

st.title("💼 Career Knowledge Assistant")
st.caption("Local, private RAG chat over resumes, interviews, negotiation & career growth guides — powered by Ollama, runs fully offline.")

if not CHROMA_PATH.exists():
    st.error("No knowledge base found. Run **Ollama.ipynb** top to bottom first to build `chroma_db/`.")
    st.stop()

try:
    _, _, all_texts, _, _, _, _ = load_backend()
except Exception as exc:
    st.error(f"Could not load the knowledge base: {exc}\n\nRun **Ollama.ipynb** top to bottom first.")
    st.stop()

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.metric("Indexed chunks", len(all_texts))
    st.caption(f"Model: `{OLLAMA_MODEL}` · Hybrid search (vector + BM25)")
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
            result = generate_rag_response(prompt, history)
            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander(f"📎 {len(result['sources'])} source(s) used"):
                    for s in result["sources"]:
                        st.caption(
                            f"**{s['file_name']}** — page {s['page_number']} "
                            f"· vector {s['vector_sim']}% · bm25 {s['bm25_score']}"
                        )

            if result["citation_warnings"]:
                st.warning("⚠️ Unverifiable citation(s): " + "; ".join(result["citation_warnings"]))

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
