"""Render PDF pages and converted DOCX/PPTX pages as PNG images."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from .discovery import discover_documents
from .schemas import DocumentSource, RenderedPage


class DocumentConversionError(RuntimeError):
    """Raised when a source document cannot be converted or rendered."""


_COMMON_MACOS_SOFFICE_PATHS = (
    Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
    Path("~/Applications/LibreOffice.app/Contents/MacOS/soffice"),
)


def _resolve_soffice_binary() -> str:
    """Resolve a usable LibreOffice executable across Linux and macOS."""

    configured = os.getenv("SOFFICE_BINARY", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and os.access(configured_path, os.X_OK):
            return str(configured_path.resolve())
        discovered = shutil.which(configured)
        if discovered:
            return discovered
        raise DocumentConversionError(
            "The LibreOffice executable configured by SOFFICE_BINARY could not be found or "
            f"is not executable: {configured}. Set SOFFICE_BINARY to a valid executable path."
        )

    for executable_name in ("soffice", "libreoffice"):
        discovered = shutil.which(executable_name)
        if discovered:
            return discovered

    for candidate in _COMMON_MACOS_SOFFICE_PATHS:
        expanded_candidate = candidate.expanduser()
        if expanded_candidate.is_file() and os.access(expanded_candidate, os.X_OK):
            return str(expanded_candidate.resolve())

    raise DocumentConversionError(
        "LibreOffice could not be found. Install LibreOffice and ensure 'soffice' is on PATH, "
        "or set SOFFICE_BINARY to its executable path. On macOS, the standard path is "
        "/Applications/LibreOffice.app/Contents/MacOS/soffice."
    )


def render_document(
    source: DocumentSource,
    *,
    page_image_dir: Path,
    processing_dir: Path,
    dpi: int,
) -> list[RenderedPage]:
    """Render one source into deterministic page or slide PNG artifacts."""

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero.")
    pdf_path = source.path
    if source.document_type != "pdf":
        pdf_path = _convert_with_libreoffice(source, processing_dir / source.document_id)
    return _render_pdf(
        pdf_path,
        source=source,
        output_dir=page_image_dir / source.document_id,
        dpi=dpi,
    )


def render_documents(
    sources: list[DocumentSource],
    *,
    page_image_dir: Path,
    processing_dir: Path,
    dpi: int,
) -> dict[str, list[RenderedPage]]:
    """Render multiple sources while retaining document boundaries."""

    return {
        source.document_id: render_document(
            source,
            page_image_dir=page_image_dir,
            processing_dir=processing_dir,
            dpi=dpi,
        )
        for source in sources
    }


def _convert_with_libreoffice(source: DocumentSource, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{source.path.stem}.pdf"
    if output_path.exists() and output_path.stat().st_mtime_ns >= source.path.stat().st_mtime_ns:
        return output_path

    profile_dir = (target_dir / "libreoffice-profile").resolve()
    home_dir = (target_dir / "home").resolve()
    xdg_config_dir = (target_dir / "xdg-config").resolve()
    for directory in (profile_dir, home_dir, xdg_config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        _resolve_soffice_binary(),
        f"-env:UserInstallation={profile_dir.as_uri()}",
        "--headless",
        "--nologo",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(target_dir),
        str(source.path),
    ]
    try:
        environment = os.environ.copy()
        environment["HOME"] = str(home_dir)
        environment["XDG_CONFIG_HOME"] = str(xdg_config_dir)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DocumentConversionError(
            f"LibreOffice conversion failed for {source.document_name}: {exc}"
        ) from exc
    if completed.returncode != 0 or not output_path.exists():
        stderr = (completed.stderr or completed.stdout or "no converter output").strip()[:1000]
        raise DocumentConversionError(
            f"LibreOffice conversion failed for {source.document_name}: {stderr}"
        )
    return output_path


def _render_pdf(
    pdf_path: Path,
    *,
    source: DocumentSource,
    output_dir: Path,
    dpi: int,
) -> list[RenderedPage]:
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[RenderedPage] = []
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise DocumentConversionError(
            f"Could not open rendered PDF for {source.document_name}: {exc}"
        ) from exc

    try:
        if document.page_count == 0:
            raise DocumentConversionError(f"Rendered document has no pages: {source.document_name}")
        for index, page in enumerate(document, start=1):
            image_path = output_dir / f"page-{index:04d}.png"
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            pixmap.save(image_path)
            image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
            location = (
                {"page_number": None, "slide_number": index}
                if source.document_type == "pptx"
                else {"page_number": index, "slide_number": None}
            )
            artifacts.append(
                RenderedPage(
                    document_id=source.document_id,
                    document_name=source.document_name,
                    document_type=source.document_type,
                    source_path=source.path,
                    source_sha256=source.source_sha256,
                    image_path=image_path,
                    image_sha256=image_sha256,
                    width=pixmap.width,
                    height=pixmap.height,
                    dpi=dpi,
                    **location,
                )
            )
    finally:
        document.close()
    return artifacts


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--docx", type=Path)
    parser.add_argument("--pptx", type=Path)
    parser.add_argument("--page-image-dir", type=Path, default=Path("data/page_images"))
    parser.add_argument("--processing-dir", type=Path, default=Path("data/processing"))
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    """Render CLI inputs and print bounded document/page counts."""

    args = _build_cli().parse_args()
    discovered = discover_documents(pdf_path=args.pdf, docx_path=args.docx, pptx_path=args.pptx)
    rendered = render_documents(
        discovered.documents,
        page_image_dir=args.page_image_dir.resolve(),
        processing_dir=args.processing_dir.resolve(),
        dpi=args.dpi,
    )
    for source in discovered.documents:
        print(f"{source.document_name}: {len(rendered[source.document_id])} rendered page(s)/slide(s)")


if __name__ == "__main__":
    main()
