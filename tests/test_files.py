"""업로드 파일 파싱 테스트. test_data/의 실제 파일을 쓴다."""

from types import SimpleNamespace

import pytest
from conftest import TEST_DATA

import app


def upload(path):
    data = path.read_bytes()
    return SimpleNamespace(name=path.name, getvalue=lambda: data)


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        ("backend/resume.pdf", "추천 API 응답 속도 개선"),
        ("backend/cover_letter.docx", "멱등키를 도입하고"),
        ("backend/portfolio.md", "요청당 쿼리 41회 → 3회"),
        ("ai_engineer/resume.md", "Recall@5를 0.61 → 0.83"),
        ("ai_engineer/cover_letter_cp949.txt", "신뢰가 핵심인 금융 도메인"),
        ("company_railpay.txt", "D+3 → D+1"),
    ],
)
def test_extract_text_from_test_data(relative_path, expected):
    path = TEST_DATA / relative_path
    text = app.extract_text(path.name, path.read_bytes())
    assert expected in text


def test_browser_pdf_spaces_are_not_control_chars():
    path = TEST_DATA / "backend/resume.pdf"
    text = app.extract_text(path.name, path.read_bytes())
    assert "\x01" not in text
    assert "백엔드 개발자" in text


def test_docx_table_rows_are_extracted():
    path = TEST_DATA / "backend/cover_letter.docx"
    text = app.extract_text(path.name, path.read_bytes())
    assert "성능 개선 | 추천 API p95 820ms → 210ms" in text


def test_bom_markdown_is_decoded():
    assert app.extract_text("a.md", "﻿# 제목".encode()) == "# 제목"


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        app.extract_text("a.hwp", b"data")


def test_edge_case_files_become_warnings():
    files = [
        upload(TEST_DATA / "edge_cases/scanned_resume.pdf"),
        upload(TEST_DATA / "edge_cases/broken.pdf"),
        upload(TEST_DATA / "edge_cases/empty.txt"),
        upload(TEST_DATA / "backend/portfolio.md"),
    ]
    materials, warnings = app.read_uploaded_files(files)
    assert [m["name"] for m in materials] == ["portfolio.md"]
    assert len(warnings) == 3
    assert warnings[0].startswith("scanned_resume.pdf: 글자를 찾지 못했어요")
    assert warnings[1].startswith("broken.pdf: 파일을 읽지 못했어요")
    assert warnings[2] == "empty.txt: 내용이 비어 있어요"
