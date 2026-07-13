"""Agentic RAG graph package for the THY case study."""

from .adapters import (
    FakeEmbeddingAdapter,
    FakeHybridRetrievalAdapter,
    FakeLlmAdapter,
    FakeGroundingEvaluatorAdapter,
    HeuristicRerankerAdapter,
    NoOpRerankerAdapter,
    OpenAIGroundingEvaluatorAdapter,
    OpenAIQueryRewriterAdapter,
    QdrantHybridRetrievalAdapter,
    RagAdapters,
    create_fake_adapters,
    create_openai_qdrant_adapters,
)
from .graphs import build_rag_graph
from .fallbacks import (
    GENERAL_KNOWLEDGE_NOTICE,
    FallbackDecision,
    assess_general_fallback_eligibility,
    compose_general_knowledge_answer,
    compose_hybrid_answer,
)
from .rerankers import OpenAIRerankerAdapter
from .runner import run_rag_question, run_rag_question_sync
from .schemas import (
    ConversationTurn,
    OpenAIRerankCandidate,
    OpenAIRerankResult,
    QueryVariant,
    RagRequest,
    RagResponse,
    RagState,
    RerankAdapterResult,
    RetrievalQueryRewrite,
    VariantRetrievalResult,
)
from .settings import RagSettings

__all__ = [
    "FakeEmbeddingAdapter",
    "FakeHybridRetrievalAdapter",
    "FakeLlmAdapter",
    "FakeGroundingEvaluatorAdapter",
    "FallbackDecision",
    "GENERAL_KNOWLEDGE_NOTICE",
    "HeuristicRerankerAdapter",
    "NoOpRerankerAdapter",
    "OpenAIGroundingEvaluatorAdapter",
    "OpenAIQueryRewriterAdapter",
    "OpenAIRerankerAdapter",
    "OpenAIRerankCandidate",
    "OpenAIRerankResult",
    "QdrantHybridRetrievalAdapter",
    "RagAdapters",
    "ConversationTurn",
    "QueryVariant",
    "RagRequest",
    "RagResponse",
    "RagSettings",
    "RagState",
    "RerankAdapterResult",
    "RetrievalQueryRewrite",
    "VariantRetrievalResult",
    "build_rag_graph",
    "assess_general_fallback_eligibility",
    "compose_general_knowledge_answer",
    "compose_hybrid_answer",
    "create_fake_adapters",
    "create_openai_qdrant_adapters",
    "run_rag_question",
    "run_rag_question_sync",
]
