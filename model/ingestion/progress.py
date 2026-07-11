"""Shared asynchronous progress reporting for document ingestion pipelines."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable


ProgressCallback = Callable[[int, int], Awaitable[None] | None]


async def report_progress(
    callback: ProgressCallback | None,
    *,
    total_pages: int,
    processed_pages: int,
) -> None:
    """Report one bounded, page-based progress snapshot when configured."""

    if callback is None:
        return
    bounded_total = max(0, total_pages)
    bounded_processed = min(max(0, processed_pages), bounded_total)
    result = callback(bounded_total, bounded_processed)
    if inspect.isawaitable(result):
        await result
