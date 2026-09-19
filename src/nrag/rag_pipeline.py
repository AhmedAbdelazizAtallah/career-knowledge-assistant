import logging
from dataclasses import dataclass

from nrag.generation.citation_check import find_unverifiable_citations
from nrag.generation.ollama_client import OllamaClient
from nrag.generation.prompt import SYSTEM_INSTRUCTION, build_user_message
from nrag.retrieval.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: list[RetrievedChunk]
    unverifiable_citations: list[str]


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: OllamaClient) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(self, query: str, top_k: int) -> RagResponse:
        retrieved = self._retriever.retrieve(query, top_k=top_k)

        if not retrieved:
            return RagResponse(
                answer="No relevant information was found in the knowledge base for this question.",
                sources=[],
                unverifiable_citations=[],
            )

        user_message = build_user_message(query, retrieved)
        answer = self._generator.chat(SYSTEM_INSTRUCTION, user_message)

        problems = find_unverifiable_citations(answer, retrieved)
        if problems:
            logger.warning("Model produced unverifiable citation(s): %s", problems)

        return RagResponse(answer=answer, sources=retrieved, unverifiable_citations=problems)
