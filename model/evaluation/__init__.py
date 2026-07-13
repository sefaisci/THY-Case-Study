"""Public contracts for deterministic, offline RAG evaluation."""

from .rag_metrics import (
    RAG_EVAL_CASE_SCHEMA_VERSION,
    RAG_EVAL_METRICS_SCHEMA_VERSION,
    RAG_EVAL_PREDICTION_SCHEMA_VERSION,
    RagEvalCase,
    RagEvalMetrics,
    RagEvalPrediction,
    evaluate_predictions,
)

__all__ = [
    "RAG_EVAL_CASE_SCHEMA_VERSION",
    "RAG_EVAL_METRICS_SCHEMA_VERSION",
    "RAG_EVAL_PREDICTION_SCHEMA_VERSION",
    "RagEvalCase",
    "RagEvalMetrics",
    "RagEvalPrediction",
    "evaluate_predictions",
]
