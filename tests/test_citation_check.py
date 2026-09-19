from nrag.generation.citation_check import find_unverifiable_citations
from nrag.retrieval.retriever import RetrievedChunk


def _chunk(file_name="resume.pdf", page=2):
    return RetrievedChunk(text="...", file_name=file_name, page_number=page, similarity=90.0)


def test_valid_citation_is_not_flagged():
    answer = "Use the X-Y-Z formula. (resume.pdf, Page 2)"
    assert find_unverifiable_citations(answer, [_chunk()]) == []


def test_citation_to_unretrieved_page_is_flagged():
    answer = "Some claim. (resume.pdf, Page 9)"
    problems = find_unverifiable_citations(answer, [_chunk(page=2)])
    assert any("Page 9" in p for p in problems)


def test_placeholder_citation_is_flagged():
    answer = "Some claim, as noted in Document [1]. (resume.pdf, Page 2)"
    problems = find_unverifiable_citations(answer, [_chunk()])
    assert any("placeholder" in p for p in problems)


def test_no_citations_means_no_problems():
    assert find_unverifiable_citations("No citation here.", [_chunk()]) == []
