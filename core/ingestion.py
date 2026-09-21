"""
core/ingestion.py -- PDF parsing, structure-aware chunking, and metadata
extraction. No Cohere calls happen here; this module only produces Chunk
objects for core/retriever.py to embed and index.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 700
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 40
MIN_PAGE_CHARS = 30

_DISCLAIMER_PATTERN = re.compile(
    r"This guide reflects widely-observed hiring practices.*?industry, and location\.",
    re.DOTALL | re.IGNORECASE,
)
_SECTION_HEADER_PATTERN = re.compile(r"^(\d+\.\s+.+)$", re.MULTILINE)


@dataclass(frozen=True)
class PageRecord:
    file_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_name: str
    document_title: str
    section: str  # "" if the chunk falls before any numbered header
    page_number: int
    text: str


def extract_text_from_pdfs(folder_path: Path) -> list[PageRecord]:
    records: list[PageRecord] = []
    pdf_files = sorted(
        f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"
    )
    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                records.append(PageRecord(pdf_path.name, page_num, text))
    return records


def clean_document_text(text: str) -> str:
    text = _DISCLAIMER_PATTERN.sub("", text)
    text = text.replace("\x7f", "")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # rejoin hyphenated line breaks
    text = re.sub(r"(?:[\r\n]+\s*)+[■•▪►*\-]\s*", r"\n- ", text)  # normalize bullets
    text = re.sub(r"\n[-*]\s*\n", "\n- ", text)
    text = re.sub(r"\n(\d+\.\s)", r"\n\n\1", text)  # break before numbered sections
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def derive_document_titles(records: list[PageRecord]) -> dict[str, str]:
    """Use each PDF's own first line of page 1 as its title -- richer and
    more human-readable than the raw filename, falls back to the filename
    if page 1 is missing or empty for some reason."""
    titles: dict[str, str] = {}
    for r in records:
        if r.page_number == 1 and r.file_name not in titles:
            first_line = r.text.strip().split("\n")[0].strip()
            titles[r.file_name] = first_line or r.file_name
    for r in records:
        titles.setdefault(r.file_name, r.file_name)
    return titles


def split_into_sections(cleaned_text: str) -> list[tuple[str, str]]:
    """Split a cleaned page's text into (section_header, section_text)
    blocks at numbered headers (e.g. "1. Format & Structure"), so chunking
    below can respect real document structure instead of cutting blindly
    by character count. Text before the first header (if any) gets an
    empty-string header."""
    matches = list(_SECTION_HEADER_PATTERN.finditer(cleaned_text))
    if not matches:
        return [("", cleaned_text)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        lead = cleaned_text[: matches[0].start()].strip()
        if lead:
            sections.append(("", lead))

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        body = cleaned_text[start:end].strip()
        sections.append((header, f"{header}\n{body}" if body else header))

    return sections


def build_chunks(data_dir: Path) -> list[Chunk]:
    """Extract -> clean -> split by section -> size-split within each
    section. Chunk IDs are content hashes (stable across re-runs on
    unchanged text), and each chunk carries its document title, section
    heading, and page number as metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    records = extract_text_from_pdfs(data_dir)
    titles = derive_document_titles(records)

    chunks: list[Chunk] = []
    for page in records:
        cleaned = clean_document_text(page.text)
        if len(cleaned) <= MIN_PAGE_CHARS:
            continue
        for section_header, section_text in split_into_sections(cleaned):
            for split in splitter.split_text(section_text):
                stripped = split.strip()
                if len(stripped) <= MIN_CHUNK_CHARS:
                    continue
                digest = hashlib.sha256(
                    f"{page.file_name}|{page.page_number}|{stripped}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append(Chunk(
                    chunk_id=f"{page.file_name}_p{page.page_number}_{digest}",
                    file_name=page.file_name,
                    document_title=titles[page.file_name],
                    section=section_header,
                    page_number=page.page_number,
                    text=stripped,
                ))
    return chunks
