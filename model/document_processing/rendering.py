"""Render PDF pages and converted DOCX/PPTX pages as PNG images."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import shutil
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


async def render_document(
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
        pdf_path = await _convert_with_libreoffice(
            source,
            processing_dir / source.document_id,
        )
    return await asyncio.to_thread(
        _render_pdf,
        pdf_path,
        source=source,
        output_dir=page_image_dir / source.document_id,
        dpi=dpi,
    )


async def render_documents(
    sources: list[DocumentSource],
    *,
    page_image_dir: Path,
    processing_dir: Path,
    dpi: int,
    max_concurrency: int = 2,
) -> dict[str, list[RenderedPage]]:
    """Render multiple sources while retaining document boundaries."""

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero.")
    rendered: list[list[RenderedPage]] = []
    for start in range(0, len(sources), max_concurrency):
        source_batch = sources[start : start + max_concurrency]
        rendered.extend(
            await asyncio.gather(
                *(
                    render_document(
                        source,
                        page_image_dir=page_image_dir,
                        processing_dir=processing_dir,
                        dpi=dpi,
                    )
                    for source in source_batch
                )
            )
        )
    return {
        source.document_id: pages
        for source, pages in zip(sources, rendered, strict=True)
    }


async def _convert_with_libreoffice(source: DocumentSource, target_dir: Path) -> Path:
    """Convert an Office document with a cancellable, timeout-bounded subprocess."""

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
    environment = os.environ.copy()
    environment["HOME"] = str(home_dir)
    environment["XDG_CONFIG_HOME"] = str(xdg_config_dir)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        raise DocumentConversionError(
            f"LibreOffice conversion failed for {source.document_name}: {exc}"
        ) from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
    except TimeoutError as exc:
        await _stop_subprocess(process)
        raise DocumentConversionError(
            f"LibreOffice conversion timed out for {source.document_name}."
        ) from exc
    except asyncio.CancelledError:
        await _stop_subprocess(process)
        raise
    if process.returncode != 0 or not output_path.exists():
        output = (stderr or stdout or b"no converter output").decode(
            "utf-8",
            errors="replace",
        )
        raise DocumentConversionError(
            f"LibreOffice conversion failed for {source.document_name}: {output.strip()[:1000]}"
        )
    return output_path


async def _stop_subprocess(process: asyncio.subprocess.Process) -> None:
    """Terminate a running converter and always reap the child process."""

    if process.returncode is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return
    try:
        await process.communicate()
    except ProcessLookupError:
        return


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


async def _main() -> None:
    """Render CLI inputs and print bounded document/page counts."""

    args = _build_cli().parse_args()
    discovered = discover_documents(pdf_path=args.pdf, docx_path=args.docx, pptx_path=args.pptx)
    rendered = await render_documents(
        discovered.documents,
        page_image_dir=args.page_image_dir.resolve(),
        processing_dir=args.processing_dir.resolve(),
        dpi=args.dpi,
    )
    for source in discovered.documents:
        print(f"{source.document_name}: {len(rendered[source.document_id])} rendered page(s)/slide(s)")


def main() -> None:
    """Run the asynchronous rendering CLI from a synchronous process boundary."""

    asyncio.run(_main())


if __name__ == "__main__":
    main()
