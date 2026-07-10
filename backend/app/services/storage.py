"""Persistent upload storage with content and MIME validation."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ..exceptions import AppError


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    filename: str
    extension: str
    mime_type: str
    size_bytes: int
    sha256: str


_STANDARD_MIME_TYPES = {
    "pdf": {"application/pdf"},
    "docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    "pptx": {"application/vnd.openxmlformats-officedocument.presentationml.presentation"},
}
_GENERIC_UPLOAD_MIME_TYPES = {"application/octet-stream", "application/zip", ""}


class UploadStorage:
    def __init__(self, root: Path, *, max_size_bytes: int, allowed_extensions: set[str]) -> None:
        self.root = root
        self.max_size_bytes = max_size_bytes
        self.allowed_extensions = allowed_extensions

    def store(
        self,
        *,
        stream: BinaryIO,
        original_filename: str,
        declared_mime_type: str | None,
        user_id: str,
        document_id: str,
    ) -> StoredUpload:
        filename = self._safe_filename(original_filename)
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in self.allowed_extensions:
            raise AppError(
                "Only PDF, DOCX, and PPTX files are supported.",
                code="unsupported_file_extension",
                status_code=422,
            )
        declared = (declared_mime_type or "").lower().strip()
        if declared not in _STANDARD_MIME_TYPES[extension] | _GENERIC_UPLOAD_MIME_TYPES:
            raise AppError(
                f"The declared MIME type {declared or 'unknown'} does not match a {extension.upper()} upload.",
                code="invalid_mime_type",
                status_code=422,
            )

        pending_dir = self.root / ".pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_path = pending_dir / f"{uuid.uuid4()}.upload"
        digest = hashlib.sha256()
        size = 0
        try:
            with pending_path.open("wb") as destination:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > self.max_size_bytes:
                        raise AppError(
                            f"File exceeds the {self.max_size_bytes // (1024 * 1024)} MB upload limit.",
                            code="file_too_large",
                            status_code=413,
                        )
                    digest.update(block)
                    destination.write(block)
            if size == 0:
                raise AppError("Uploaded files must not be empty.", code="empty_file", status_code=422)
            self._validate_content(pending_path, extension)
            target_dir = self.root / user_id / document_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename
            os.replace(pending_path, target_path)
            return StoredUpload(
                path=target_path,
                filename=filename,
                extension=extension,
                mime_type=next(iter(_STANDARD_MIME_TYPES[extension])),
                size_bytes=size,
                sha256=digest.hexdigest(),
            )
        except Exception:
            pending_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def remove(path: str | Path) -> None:
        source = Path(path)
        source.unlink(missing_ok=True)
        try:
            source.parent.rmdir()
        except OSError:
            pass

    @staticmethod
    def _safe_filename(filename: str) -> str:
        basename = Path(filename or "").name
        if not basename or basename != filename:
            raise AppError("Invalid upload filename.", code="invalid_filename", status_code=422)
        sanitized = re.sub(r"[\x00-\x1f\x7f]", "", basename).strip()
        if not sanitized or len(sanitized) > 255:
            raise AppError("Invalid upload filename.", code="invalid_filename", status_code=422)
        return sanitized

    @staticmethod
    def _validate_content(path: Path, extension: str) -> None:
        if extension == "pdf":
            with path.open("rb") as source:
                signature = source.read(5)
            if signature != b"%PDF-":
                raise AppError(
                    "The uploaded file is not a valid PDF container.",
                    code="invalid_file_content",
                    status_code=422,
                )
            return
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            raise AppError(
                f"The uploaded file is not a valid {extension.upper()} container.",
                code="invalid_file_content",
                status_code=422,
            ) from exc
        required = "word/document.xml" if extension == "docx" else "ppt/presentation.xml"
        if required not in names:
            raise AppError(
                f"The uploaded file content does not match {extension.upper()}.",
                code="invalid_file_content",
                status_code=422,
            )
