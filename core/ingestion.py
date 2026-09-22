"""
core/ingestion.py -- PDF parsing, noise removal, structure-aware
parent-child chunking, and metadata extraction. No embedding or LLM calls
happen here; this module only produces Chunk objects for
core/vectorstore.py to persist and core/retriever.py to search.

Pipeline: extract (text + tables) -> strip repeated headers/footers ->
clean -> split into logical sections (carried across page breaks) ->
size-split each section into child chunks, while keeping the full
section as that child's parent context.
"""
import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 40
MIN_PAGE_CHARS = 30

# A line repeated on at least this fraction of a document's pages is
# treated as a running header/footer, not content.
REPEATED_LINE_MIN_FRACTION = 0.6
REPEATED_LINE_MAX_CHARS = 100

_DISCLAIMER_PATTERN = re.compile(
    r"This guide reflects widely-observed hiring practices.*?industry, and location\.",
    re.DOTALL | re.IGNORECASE,
)
_SECTION_HEADER_PATTERN = re.compile(r"^(\d+\.\s+.+)$", re.MULTILINE)
_PAGE_NUMBER_LINE_PATTERN = re.compile(
    r"^\s*(page\s+)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class PageRecord:
    file_name: str
    page_number: int
    text: str
    tables_markdown: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    document_title: str
    section: str  # "" only for the lead text before any numbered header
    page_number: int
    text: str
    parent_id: str
    parent_text: str
    parent_page_start: int
    parent_page_end: int


def document_id_for(file_name: str) -> str:
    """Stable id independent of the human-readable file name, so a file
    can be renamed without orphaning chunks tied to the old name."""
    return hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:12]


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [
        [("" if cell is None else str(cell).strip().replace("\n", " ")) for cell in row]
        for row in rows
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if len(cleaned) < 2:
        return ""
    header, *body = cleaned
    ncols = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    for row in body:
        row = (row + [""] * ncols)[:ncols]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_tables_by_page(pdf_path: Path) -> dict[int, tuple[str, ...]]:
    """Best-effort table extraction via pdfplumber, run independently of
    the main pypdf text extraction below. A page with unparsable tables
    (or a PDF pdfplumber can't open at all) simply yields no tables --
    this must never abort ingestion of the rest of the document."""
    tables_by_page: dict[int, tuple[str, ...]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                except Exception as exc:
                    logger.warning("Table extraction failed on %s page %d: %s", pdf_path.name, i, exc)
                    continue
                rendered = tuple(md for t in tables if (md := _table_to_markdown(t)))
                if rendered:
                    tables_by_page[i] = rendered
    except Exception as exc:
        logger.warning("pdfplumber could not open %s for table extraction: %s", pdf_path.name, exc)
    return tables_by_page


def extract_text_from_pdfs(folder_path: Path) -> list[PageRecord]:
    records: list[PageRecord] = []
    pdf_files = sorted(
        f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"
    )
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
        except (PdfReadError, OSError) as exc:
            logger.error("Skipping malformed PDF %s: %s", pdf_path.name, exc)
            continue

        tables_by_page = _extract_tables_by_page(pdf_path)

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:
                logger.warning("Skipping unreadable page %d of %s: %s", page_num, pdf_path.name, exc)
                continue
            tables = tables_by_page.get(page_num, ())
            if text or tables:
                records.append(PageRecord(pdf_path.name, page_num, text, tables))
    return records


def _strip_repeated_lines(records: list[PageRecord]) -> list[PageRecord]:
    """Remove running headers/footers and bare page-number lines: any
    short line repeated across most of a document's pages is noise, not
    content, and left in place it pollutes every chunk's embedding with
    identical boilerplate."""
    by_file: dict[str, list[PageRecord]] = defaultdict(list)
    for r in records:
        by_file[r.file_name].append(r)

    cleaned: list[PageRecord] = []
    for file_name, pages in by_file.items():
        repeated: set[str] = set()
        if len(pages) >= 3:
            counts = Counter()
            for p in pages:
                for line in {ln.strip() for ln in p.text.splitlines() if ln.strip()}:
                    if len(line) <= REPEATED_LINE_MAX_CHARS:
                        counts[line] += 1
            threshold = max(2, round(len(pages) * REPEATED_LINE_MIN_FRACTION))
            repeated = {line for line, count in counts.items() if count >= threshold}

        for p in pages:
            kept_lines = [
                ln for ln in p.text.splitlines()
                if ln.strip() not in repeated and not _PAGE_NUMBER_LINE_PATTERN.match(ln)
            ]
            cleaned.append(PageRecord(p.file_name, p.page_number, "\n".join(kept_lines), p.tables_markdown))
    return cleaned


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


def split_into_sections(
    cleaned_text: str, carry_in_header: str = ""
) -> tuple[list[tuple[str, str]], str]:
    """Split a cleaned page's text into (section_header, section_text)
    blocks at numbered headers (e.g. "2. Format & Structure"), so chunking
    below can respect real document structure instead of cutting blindly
    by character count.

    `carry_in_header` is the section still open when the previous page
    ended. Sections routinely span a page break with no new header on the
    continuation page -- without carrying it in, that continuation text
    would silently lose its section label (breaking citations) and its
    embedded text would lose the heading context that ties it back to the
    section (weakening retrieval), even though it's still part of that
    section. It's also prepended into the chunk text itself so the
    embedding carries that heading context, the same way a chunk that
    starts with its own header does.

    Returns (sections, last_header_on_this_page) so the caller can pass
    last_header_on_this_page as the next page's carry_in_header.
    """
    matches = list(_SECTION_HEADER_PATTERN.finditer(cleaned_text))

    sections: list[tuple[str, str]] = []
    lead_end = matches[0].start() if matches else len(cleaned_text)
    lead = cleaned_text[:lead_end].strip()
    if lead:
        text = f"{carry_in_header}\n{lead}" if carry_in_header else lead
        sections.append((carry_in_header, text))

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        body = cleaned_text[start:end].strip()
        sections.append((header, f"{header}\n{body}" if body else header))

    last_header = matches[-1].group(1).strip() if matches else carry_in_header
    return sections, last_header


@dataclass
class _SectionInstance:
    document_id: str
    document_name: str
    document_title: str
    section: str
    page_number: int
    text: str


def _parent_key(inst: _SectionInstance) -> str:
    return f"{inst.document_id}::{inst.section or f'(untitled p{inst.page_number})'}"


def _build_section_instances(data_dir: Path) -> list[_SectionInstance]:
    records = extract_text_from_pdfs(data_dir)
    records = _strip_repeated_lines(records)
    titles = derive_document_titles(records)

    instances: list[_SectionInstance] = []
    open_section: dict[str, str] = {}  # file_name -> section still open at page end
    for page in records:
        cleaned = clean_document_text(page.text)
        if page.tables_markdown:
            cleaned = (cleaned + "\n\nTable (extracted):\n" + "\n\nTable (extracted):\n".join(page.tables_markdown)).strip()
        if len(cleaned) <= MIN_PAGE_CHARS:
            continue

        carry_in = open_section.get(page.file_name, "")
        sections, last_header = split_into_sections(cleaned, carry_in)
        open_section[page.file_name] = last_header

        doc_id = document_id_for(page.file_name)
        for section_header, section_text in sections:
            instances.append(_SectionInstance(
                document_id=doc_id, document_name=page.file_name,
                document_title=titles[page.file_name], section=section_header,
                page_number=page.page_number, text=section_text,
            ))
    return instances


def _aggregate_parents(
    instances: list[_SectionInstance],
) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    """One logical section can be spread across several pages (each its
    own `_SectionInstance` sharing the same header). This concatenates
    them in document order into a single parent text per section, so a
    child chunk retrieved from anywhere in that section can be expanded
    to the section's full content -- not just the page-sized fragment it
    happened to be embedded from. Also returns each parent's (first,
    last) page number, since a caller citing the expanded parent text
    needs the real page range it spans, not just one child's page."""
    by_key: dict[str, list[_SectionInstance]] = defaultdict(list)
    for inst in instances:
        by_key[_parent_key(inst)].append(inst)

    parent_texts: dict[str, str] = {}
    parent_pages: dict[str, tuple[int, int]] = {}
    for key, group in by_key.items():
        group.sort(key=lambda i: i.page_number)
        pieces: list[str] = []
        header = group[0].section
        for inst in group:
            piece = inst.text
            if header and piece.startswith(header):
                piece = piece[len(header):].lstrip("\n")
            if piece:
                pieces.append(piece)
        body = "\n\n".join(pieces)
        parent_texts[key] = f"{header}\n{body}" if header else body
        parent_pages[key] = (group[0].page_number, group[-1].page_number)
    return parent_texts, parent_pages


def build_chunks(data_dir: Path) -> list[Chunk]:
    """Extract -> clean -> split by section (carried across pages) ->
    size-split within each section into child chunks. Each child keeps a
    link (`parent_id`) to its full section text (`parent_text`) for
    parent-child context expansion at answer time. Chunk ids are content
    hashes, stable across re-runs on unchanged text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    instances = _build_section_instances(data_dir)
    parent_texts, parent_pages = _aggregate_parents(instances)

    chunks: list[Chunk] = []
    for inst in instances:
        parent_id = _parent_key(inst)
        page_start, page_end = parent_pages[parent_id]
        for split in splitter.split_text(inst.text):
            stripped = split.strip()
            if len(stripped) <= MIN_CHUNK_CHARS:
                continue
            digest = hashlib.sha256(
                f"{inst.document_name}|{inst.page_number}|{stripped}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=f"{inst.document_name}_p{inst.page_number}_{digest}",
                document_id=inst.document_id,
                document_name=inst.document_name,
                document_title=inst.document_title,
                section=inst.section,
                page_number=inst.page_number,
                text=stripped,
                parent_id=parent_id,
                parent_text=parent_texts[parent_id],
                parent_page_start=page_start,
                parent_page_end=page_end,
            ))
    return chunks
