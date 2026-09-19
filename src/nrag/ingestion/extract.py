import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageRecord:
    file_name: str
    file_path: str
    page_number: int
    total_pages: int
    text: str


def extract_text_from_pdfs(folder_path: Path) -> list[PageRecord]:
    """Extract raw per-page text from every PDF in `folder_path`."""
    records: list[PageRecord] = []

    pdf_files = sorted(
        f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        logger.warning("No PDF files found in %s", folder_path)
        return records

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                cleaned = (page.extract_text() or "").strip()
                if cleaned:
                    records.append(
                        PageRecord(
                            file_name=pdf_path.name,
                            file_path=str(pdf_path),
                            page_number=page_num,
                            total_pages=total_pages,
                            text=cleaned,
                        )
                    )
            logger.info("Processed %s (%d pages)", pdf_path.name, total_pages)
        except Exception:
            logger.exception("Failed to read %s", pdf_path.name)

    return records
