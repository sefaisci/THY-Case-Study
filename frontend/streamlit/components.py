"""Focused Streamlit presentation helpers for citations, usage, and documents."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    st.markdown("#### Citations")
    for citation in citations:
        location = (
            f"page {citation['page_number']}"
            if citation.get("page_number") is not None
            else f"slide {citation.get('slide_number')}"
        )
        excerpt = html.escape(str(citation.get("source_excerpt", "")))
        st.markdown(
            (
                '<div class="thy-citation">'
                f"<strong>{html.escape(str(citation.get('filename', 'Unknown file')))}</strong>"
                '<div class="thy-citation-meta">'
                f"{html.escape(location)} · score {float(citation.get('retrieval_score', 0.0)):.3f} · "
                f"{html.escape(str(citation.get('ingestion_method', '')))} · "
                f"{html.escape(str(citation.get('source_collection', '')))}"
                "</div>"
                f'<div class="thy-citation-excerpt">{excerpt}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_usage_details(
    request_usage: dict[str, Any] | None,
    session_usage: dict[str, Any] | None,
    total_usage: dict[str, Any] | None,
    *,
    model: str | None,
    reasoning_effort: str | None,
) -> None:
    if not any((request_usage, session_usage, total_usage)):
        return
    with st.expander("Token & cost details", expanded=False):
        if model:
            st.caption(f"Model: {model} · Reasoning effort: {reasoning_effort or 'not applicable'}")
        columns = st.columns(3)
        for column, label, usage in zip(
            columns,
            ("Request", "Session", "Total"),
            (request_usage, session_usage, total_usage),
            strict=True,
        ):
            with column:
                st.markdown(f"**{label}**")
                if not usage:
                    st.caption("Not available")
                    continue
                st.metric("Tokens", f"{int(usage.get('total_tokens', 0)):,}")
                st.caption(
                    f"Input {int(usage.get('input_tokens', 0)):,} · "
                    f"Cached {int(usage.get('cached_input_tokens', 0)):,} · "
                    f"Output {int(usage.get('output_tokens', 0)):,} · "
                    f"Reasoning {int(usage.get('reasoning_tokens', 0)):,}"
                )
                cost = float(usage.get("cost_usd", 0.0))
                st.metric("Known cost", f"${cost:.6f}")
                if int(usage.get("unpriced_record_count", 0)):
                    st.caption("Cost is incomplete because at least one model is unpriced.")


def document_summary(document: dict[str, Any]) -> str:
    status = html.escape(str(document.get("status", "unknown")))
    semantic = ""
    if document.get("semantic_model"):
        semantic = (
            f" · {html.escape(str(document['semantic_model']))} / "
            f"{html.escape(str(document.get('semantic_reasoning_effort') or ''))}"
        )
    return (
        '<div class="thy-document-row">'
        f'<div class="thy-document-name">{html.escape(str(document.get("filename", "")))}</div>'
        '<div class="thy-document-meta">'
        f'<span class="thy-status {status}"></span>{status} · '
        f"{html.escape(str(document.get('file_extension', '')).upper())} · "
        f"{html.escape(str(document.get('ingestion_method', '')))} · "
        f"{html.escape(str(document.get('collection_name', '')))}{semantic}"
        "</div></div>"
    )
