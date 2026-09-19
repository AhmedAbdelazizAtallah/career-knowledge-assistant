import re

from nrag.retrieval.retriever import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\(([^,()]+\.pdf),\s*Page\s*(\d+)\)", re.IGNORECASE)
_PLACEHOLDER_PATTERN = re.compile(r"\bDocument\s*\[\d+\]", re.IGNORECASE)


def find_unverifiable_citations(answer: str, retrieved: list[RetrievedChunk]) -> list[str]:
    """Return citation strings in `answer` that don't match any retrieved (file, page).

    Catches model hallucination like citing a page that wasn't actually
    retrieved, or falling back to a "Document [1]" placeholder instead of
    the real file name.
    """
    valid_pairs = {(c.file_name.lower(), c.page_number) for c in retrieved}
    problems: list[str] = []

    for match in _CITATION_PATTERN.finditer(answer):
        file_name, page = match.group(1).strip().lower(), int(match.group(2))
        if (file_name, page) not in valid_pairs:
            problems.append(match.group(0))

    if _PLACEHOLDER_PATTERN.search(answer):
        problems.append("placeholder citation (e.g. 'Document [1]') instead of a real file name")

    return problems
