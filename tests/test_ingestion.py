"""Regression tests for core/ingestion.py's structure-aware chunking.

Covers the page-boundary bug where a section that continues onto a new
page with no header of its own lost its section label and its embedded
text lost the heading context tying it back to that section.
"""
from pathlib import Path

from core.ingestion import build_chunks, split_into_sections

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_split_into_sections_carries_open_section_across_pages():
    page1 = "1. First Section\nSome intro text."
    page2 = "More text with no header on this page at all."

    sections1, last_header = split_into_sections(page1)
    assert last_header == "1. First Section"

    sections2, last_header2 = split_into_sections(page2, carry_in_header=last_header)
    assert len(sections2) == 1
    header, text = sections2[0]
    assert header == "1. First Section"
    assert text.startswith("1. First Section\n")
    assert "More text with no header" in text
    assert last_header2 == "1. First Section"


def test_split_into_sections_new_header_on_continuation_page_overrides_carry_in():
    sections, last_header = split_into_sections(
        "tail of the previous section\n2. New Section\nbody",
        carry_in_header="1. First Section",
    )
    headers = [h for h, _ in sections]
    assert headers == ["1. First Section", "2. New Section"]
    assert last_header == "2. New Section"


def test_build_chunks_has_no_unlabeled_mid_document_sections():
    """Only the page-1 title/metadata block (before any numbered header)
    should ever have a blank section label -- everything after the first
    header in a document must inherit a real section, even across pages."""
    chunks = build_chunks(DATA_DIR)
    for c in chunks:
        if c.section == "":
            assert c.page_number == 1, (
                f"{c.document_name} page {c.page_number} has a blank section "
                "past page 1 -- a section label was lost across a page break"
            )


def test_resume_format_and_structure_section_includes_full_recommended_structure():
    chunks = build_chunks(DATA_DIR)
    section_text = "\n".join(
        c.text for c in chunks
        if c.document_name == "01_Resume_Writing_Best_Practices.pdf"
        and c.section == "2. Format & Structure"
    )
    for expected in (
        "Core Competencies", "Professional Experience",
        "Education & Certifications", "Projects / Portfolio",
    ):
        assert expected in section_text


def test_parent_text_spans_full_section_across_pages():
    """Section 2 spans page 1 and page 2 in the source PDF -- every child
    chunk from that section should point at one shared parent_text
    containing both pages' content, not just its own page fragment."""
    chunks = build_chunks(DATA_DIR)
    resume_ch2 = [
        c for c in chunks
        if c.document_name == "01_Resume_Writing_Best_Practices.pdf"
        and c.section == "2. Format & Structure"
    ]
    assert resume_ch2
    parent_texts = {c.parent_text for c in resume_ch2}
    assert len(parent_texts) == 1, "all children of one section should share one parent text"
    parent_text = next(iter(parent_texts))
    assert "Layout rules" in parent_text  # page 1 content
    assert "Core Competencies" in parent_text  # page 2 content


def test_document_id_stable_and_unique_per_file():
    chunks = build_chunks(DATA_DIR)
    ids_by_name = {c.document_name: c.document_id for c in chunks}
    assert len(set(ids_by_name.values())) == len(ids_by_name)
