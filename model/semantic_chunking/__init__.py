"""Strict, flat page-image semantic chunking."""

from .openai_adapter import OpenAISemanticChunker, SemanticChunkingError, image_to_data_url
from .pipeline import SemanticChunkingPipeline
from .schemas import SemanticChunk, SemanticPageResult

__all__ = [
    "OpenAISemanticChunker",
    "SemanticChunk",
    "SemanticChunkingError",
    "SemanticChunkingPipeline",
    "SemanticPageResult",
    "image_to_data_url",
]
