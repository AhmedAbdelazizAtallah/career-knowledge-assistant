from nrag.ingestion.clean import clean_document_text


def test_removes_disclaimer_boilerplate():
    text = (
        "Real content before.\n"
        "This guide reflects widely-observed hiring practices and may vary by company, "
        "industry, and location.\n"
        "Real content after."
    )
    result = clean_document_text(text)
    assert "widely-observed hiring practices" not in result
    assert "Real content before." in result
    assert "Real content after." in result


def test_strips_control_characters():
    assert "\x7f" not in clean_document_text("Some\x7f text\x7f here")


def test_rejoins_hyphenated_line_breaks():
    assert clean_document_text("respon-\nsible") == "responsible"


def test_normalizes_bullet_markers_to_dash():
    text = "Intro\n■ First point\n• Second point\n▪ Third point"
    result = clean_document_text(text)
    assert "■" not in result and "•" not in result and "▪" not in result
    assert "- First point" in result
    assert "- Second point" in result


def test_collapses_excess_blank_lines_and_spaces():
    result = clean_document_text("a   b\n\n\n\nc")
    assert "   " not in result
    assert "\n\n\n" not in result
