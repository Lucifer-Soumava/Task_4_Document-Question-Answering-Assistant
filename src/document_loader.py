"""Local document ingestion for PDF, DOCX, TXT, and Markdown files."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Any


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(filename: str, content: bytes) -> list[dict[str, Any]]:
    """Extract text and retain the source filename and page/paragraph location."""
    suffix = Path(filename).suffix.lower()
    segments: list[dict[str, Any]] = []

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to read PDF files: pip install pypdf") from exc
        reader = PdfReader(BytesIO(content))
        for page_number, page in enumerate(reader.pages, start=1):
            text = _clean_text(page.extract_text() or "")
            if text:
                # Keep the page as one source segment; chunking will preserve page boundaries.
                segments.append({
                    "source": filename, "page": page_number,
                    "paragraph": 1, "text": text,
                })

    elif suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Install python-docx to read DOCX files: pip install python-docx") from exc
        document = Document(BytesIO(content))
        paragraph_number = 0
        for paragraph in document.paragraphs:
            text = _clean_text(paragraph.text)
            if text:
                paragraph_number += 1
                segments.append({
                    "source": filename, "page": None,
                    "paragraph": paragraph_number, "text": text,
                })
        for table in document.tables:
            for row in table.rows:
                text = _clean_text(" | ".join(cell.text for cell in row.cells))
                if text:
                    paragraph_number += 1
                    segments.append({
                        "source": filename, "page": None,
                        "paragraph": paragraph_number, "text": text,
                    })

    elif suffix in {".txt", ".md"}:
        try:
            raw_text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raw_text = content.decode("cp1252", errors="replace")
        paragraphs = [
            _clean_text(part) for part in re.split(r"\n\s*\n", raw_text)
            if _clean_text(part)
        ]
        for paragraph_number, paragraph in enumerate(paragraphs, start=1):
            segments.append({
                "source": filename, "page": None,
                "paragraph": paragraph_number, "text": paragraph,
            })
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Upload PDF, DOCX, TXT, or MD files."
        )

    if not segments:
        raise ValueError(
            f"No readable text found in '{filename}'. It may be empty, image-only, or scanned."
        )
    return segments


def load_folder(folder: str | Path) -> list[dict[str, Any]]:
    """Load supported documents from a folder, skipping unsupported files."""
    folder_path = Path(folder)
    segments: list[dict[str, Any]] = []
    if not folder_path.exists():
        return segments
    for path in sorted(folder_path.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
            segments.extend(load_document(path.name, path.read_bytes()))
    return segments
