"""Optional-path document discovery with stable content identities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from pathlib import Path
from typing import TypeAlias

from .schemas import DiscoveredInputs, DocumentSource, DocumentType

PathLike: TypeAlias = str | Path
_FORMAT_ORDER: tuple[tuple[DocumentType, str], ...] = (
    ("pdf", ".pdf"),
    ("docx", ".docx"),
    ("pptx", ".pptx"),
)


def discover_documents(
    *,
    pdf_path: PathLike | None = None,
    docx_path: PathLike | None = None,
    pptx_path: PathLike | None = None,
    filename_allowlist: Collection[str] | None = None,
) -> DiscoveredInputs:
    """Discover matching files in deterministic PDF, DOCX, PPTX order.

    A supplied path may identify one matching file or a directory. Directories
    are scanned non-recursively for the extension associated with that input.
    ``None`` and empty directories are valid and produce no documents.
    """

    raw_paths: dict[DocumentType, PathLike | None] = {
        "pdf": pdf_path,
        "docx": docx_path,
        "pptx": pptx_path,
    }
    allowed = {name.casefold() for name in filename_allowlist} if filename_allowlist else None
    supplied: dict[DocumentType, Path | None] = {}
    documents: list[DocumentSource] = []

    for document_type, extension in _FORMAT_ORDER:
        raw_path = raw_paths[document_type]
        supplied[document_type] = Path(raw_path).expanduser().resolve() if raw_path is not None else None
        for path in _matching_files(raw_path, extension):
            if allowed is not None and path.name.casefold() not in allowed:
                continue
            digest = _sha256_file(path)
            safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem).strip("-").lower()
            safe_stem = safe_stem or "document"
            documents.append(
                DocumentSource(
                    path=path,
                    document_id=f"{safe_stem}-{digest[:16]}",
                    document_name=path.name,
                    document_type=document_type,
                    source_sha256=digest,
                )
            )

    return DiscoveredInputs(documents=documents, supplied_paths=supplied)


def _matching_files(raw_path: PathLike | None, extension: str) -> list[Path]:
    if raw_path is None:
        return []
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document input path does not exist: {path}")
    if path.is_file():
        if path.suffix.casefold() != extension:
            raise ValueError(f"Expected a {extension} file for input path: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Document input path is neither a file nor directory: {path}")
    return sorted(
        (candidate.resolve() for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.casefold() == extension),
        key=lambda candidate: (candidate.name.casefold(), str(candidate)),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
