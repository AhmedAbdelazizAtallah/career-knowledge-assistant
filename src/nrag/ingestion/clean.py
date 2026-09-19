import re

from nrag.ingestion.extract import PageRecord

_DISCLAIMER_PATTERN = re.compile(
    r"This guide reflects widely-observed hiring practices.*?industry, and location\.",
    re.DOTALL | re.IGNORECASE,
)
_HYPHEN_LINEBREAK = re.compile(r"(\w+)-\n(\w+)")
_BULLET_MARKERS = re.compile(r"(?:[\r\n]+\s*)+[■•▪►*\-]\s*")
_ORPHAN_BULLET = re.compile(r"\n[-*]\s*\n")
_EXTRA_SPACES = re.compile(r"[ \t]+")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def clean_document_text(text: str) -> str:
    """Strip boilerplate/control chars and normalize whitespace + bullets."""
    text = _DISCLAIMER_PATTERN.sub("", text)
    text = text.replace("\x7f", "")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _BULLET_MARKERS.sub("\n- ", text)
    text = _ORPHAN_BULLET.sub("\n- ", text)
    text = _EXTRA_SPACES.sub(" ", text)
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def preprocess_records(
    raw_records: list[PageRecord], min_page_chars: int = 30
) -> list[PageRecord]:
    cleaned: list[PageRecord] = []
    for record in raw_records:
        cleaned_text = clean_document_text(record.text)
        if len(cleaned_text) > min_page_chars:
            cleaned.append(
                PageRecord(
                    file_name=record.file_name,
                    file_path=record.file_path,
                    page_number=record.page_number,
                    total_pages=record.total_pages,
                    text=cleaned_text,
                )
            )
    return cleaned
