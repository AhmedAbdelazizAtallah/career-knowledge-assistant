from nrag.retrieval.retriever import RetrievedChunk

SYSTEM_INSTRUCTION = (
    "You are an expert career consultant. Answer the user's question directly and "
    "comprehensively using ONLY the provided context.\n"
    "Strict Rules:\n"
    "1. NEVER invent, extrapolate, or fabricate any examples. If an example is provided "
    "in the text, quote or adapt ONLY that exact example.\n"
    "2. Present formulas, frameworks, and their accompanying rules completely as stated.\n"
    "3. Every paragraph or piece of advice MUST end with an explicit source citation in "
    "the format: (exact_file_name.pdf, Page X), using the real file name given in the "
    "context below -- never a placeholder like 'Document [1]'.\n"
    "4. If the context does not contain enough information to answer, say so explicitly "
    "instead of guessing."
)


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"--- Source: {chunk.file_name} (Page {chunk.page_number}) ---\n{chunk.text}"
        for chunk in chunks
    ]
    return "\n\n".join(blocks)


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Context Documents:\n{build_context(chunks)}\n\n"
        f"Question: {query}\n\n"
        "Provide a structured, helpful answer based strictly on the context above:"
    )
