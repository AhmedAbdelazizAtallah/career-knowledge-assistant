import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("NRAG_API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Career Knowledge Assistant", page_icon="💼")
st.title("💼 Career Knowledge Assistant")
st.caption("Ask questions grounded in the local career knowledge base (RAG over Ollama).")

with st.sidebar:
    st.subheader("Knowledge base")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        st.success(f"API online — {health['chunks_indexed']} chunks indexed")
    except requests.RequestException:
        st.error("API unreachable. Is `uvicorn api.main:app` running?")

    if st.button("Re-index PDFs in data/"):
        with st.spinner("Ingesting..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/ingest", timeout=300)
                resp.raise_for_status()
                st.success(resp.json())
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about resumes, interviews, negotiation..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query", json={"query": user_query, "top_k": 4}, timeout=120
                )
                response.raise_for_status()
                payload = response.json()
                answer = payload["answer"]
                st.markdown(answer)

                if payload["unverifiable_citations"]:
                    st.warning(
                        "Note: the model produced citation(s) that couldn't be verified "
                        f"against retrieved sources: {payload['unverifiable_citations']}"
                    )

                with st.expander("Sources"):
                    for src in payload["sources"]:
                        st.markdown(
                            f"**{src['file_name']} (Page {src['page_number']})** "
                            f"— similarity {src['similarity']}%"
                        )
                        st.text(src["text"])

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except requests.RequestException as exc:
                error_message = f"Request failed: {exc}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
