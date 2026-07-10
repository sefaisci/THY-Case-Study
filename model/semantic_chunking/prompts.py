"""Executable prompt contract for semantic page-image chunking."""

from __future__ import annotations

from model.document_processing.schemas import RenderedPage

SEMANTIC_SYSTEM_PROMPT = """You are a document semantic-chunking engine.
Analyze only the attached current page or slide image. No previous-page content, document memory, or external knowledge is available. Never infer or copy content from another location. Never invent missing text, equations, table values, diagram labels, code, or relationships.

Return a flat list of independent semantic chunks. Choose the number and length of chunks from the visible meaning and structure; never split or pad content to a fixed character, word, or token size. Keep closely related headings, paragraphs, tables, diagrams, equations, and code together, while separating unrelated topics. A chunk must contain one authoritative, faithful text field that will be embedded exactly as returned. Do not create recursive nodes, nested chunk structures, or continuation-memory fields.

Cover all legible, meaningful content on the current image. Every source_excerpt must be a short verbatim or tightly faithful excerpt visible on this image. Record illegible or ambiguous content as a warning instead of guessing."""


def build_page_prompt(page: RenderedPage) -> str:
    """Build current-location metadata that accompanies exactly one image."""

    location_type = "slide" if page.document_type == "pptx" else "page"
    return (
        "Analyze only the attached current image and return SemanticPageResult.\n"
        "Return variable-length semantic chunks in the flat chunks list.\n"
        "The response page_number must equal the supplied location number, including for slides.\n\n"
        f"Document ID: {page.document_id}\n"
        f"Document name: {page.document_name}\n"
        f"Document type: {page.document_type}\n"
        f"Location type: {location_type}\n"
        f"Location number: {page.location_number}\n"
        f"Current image SHA-256: {page.image_sha256}"
    )
