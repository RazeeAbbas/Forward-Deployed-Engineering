from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class SourceDocument:
    text: str
    metadata: dict[str, Any]


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    with manifest_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            relative_path = str(entry["path"])
            missing = {"department", "access_tier", "date"} - entry.keys()
            if missing:
                raise ValueError(f"Manifest line {line_number} is missing: {sorted(missing)}")
            date.fromisoformat(str(entry["date"]))
            entries[relative_path] = {
                "department": str(entry["department"]),
                "access_tier": str(entry["access_tier"]),
                "date": str(entry["date"]),
            }
    return entries


def load_documents(data_dir: Path, manifest: dict[str, dict[str, str]]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for relative_path, tags in manifest.items():
        path = (data_dir / relative_path).resolve()
        if not path.is_relative_to(data_dir.resolve()):
            raise ValueError(f"Manifest path escapes data directory: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        base_metadata = {**tags, "source": relative_path}
        if suffix == ".md":
            documents.append(SourceDocument(path.read_text(encoding="utf-8"), base_metadata))
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as file:
                for row_number, row in enumerate(csv.DictReader(file), start=2):
                    text = "\n".join(f"{key}: {value}" for key, value in row.items() if value is not None)
                    documents.append(SourceDocument(text, {**base_metadata, "row": row_number}))
        elif suffix == ".pdf":
            for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(SourceDocument(text, {**base_metadata, "page": page_number}))
        else:
            raise ValueError(f"Unsupported document type: {suffix}")
    return documents


def _split_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        sentences = paragraph.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").splitlines()
        for sentence in sentences:
            sentence = sentence.strip()
            while len(sentence) > chunk_size:
                units.append(sentence[:chunk_size])
                sentence = sentence[chunk_size:]
            if sentence:
                units.append(sentence)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        prefix = current[-overlap:] if overlap and current else ""
        current = f"{prefix}\n{unit}".strip() if prefix else unit
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap :] if overlap else current[chunk_size:]
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    documents: Iterable[SourceDocument], chunk_size: int = 900, overlap: int = 120
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for index, text in enumerate(_split_text(document.text, chunk_size, overlap)):
            metadata = dict(document.metadata)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{metadata['source']}:{metadata.get('page', metadata.get('row', 0))}:{index}:{text}"))
            chunks.append({"id": chunk_id, "text": text, "metadata": metadata})
    return chunks