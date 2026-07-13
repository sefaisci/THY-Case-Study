"""Deterministic, service-free metrics for RAG retrieval and abstention."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from model.agentic_rag.schemas import CollectionScope, ConversationTurn

RAG_EVAL_CASE_SCHEMA_VERSION: Final = "1.0"
RAG_EVAL_PREDICTION_SCHEMA_VERSION: Final = "1.0"
RAG_EVAL_METRICS_SCHEMA_VERSION: Final = "rag-eval-metrics.v1"


def _normalize_nonblank(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank.")
    return normalized


def _normalize_string_list(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    return [
        _normalize_nonblank(value, field_name=f"{field_name} item")
        for value in values
    ]


class RagEvalCase(BaseModel):
    """One versioned, human-labeled retrieval/abstention case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"]
    case_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(min_length=1, max_length=200)
    collection_scope: CollectionScope = "both"
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    answerable: StrictBool
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    expected_no_answer: StrictBool
    tags: list[str] = Field(default_factory=list)

    @field_validator("case_id", "question", "user_id")
    @classmethod
    def validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _normalize_nonblank(value, field_name=info.field_name)

    @field_validator("relevant_chunk_ids")
    @classmethod
    def validate_relevant_chunk_ids(cls, values: list[str]) -> list[str]:
        normalized = _normalize_string_list(
            values,
            field_name="relevant_chunk_ids",
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("relevant_chunk_ids must be unique.")
        return normalized

    @field_validator("required_facts", "tags")
    @classmethod
    def validate_nonblank_lists(
        cls,
        values: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        return _normalize_string_list(values, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_answerability(self) -> "RagEvalCase":
        if self.expected_no_answer != (not self.answerable):
            raise ValueError(
                "expected_no_answer must be the inverse of answerable."
            )
        if self.answerable and not self.relevant_chunk_ids:
            raise ValueError(
                "Answerable cases require at least one relevant chunk ID."
            )
        if not self.answerable and self.relevant_chunk_ids:
            raise ValueError(
                "Unanswerable cases cannot define relevant chunk IDs."
            )
        if not self.answerable and self.required_facts:
            raise ValueError("Unanswerable cases cannot define required facts.")
        if not self.answerable and self.reference_answer is not None:
            raise ValueError(
                "Unanswerable cases cannot define a reference answer."
            )
        return self


class RagEvalPrediction(BaseModel):
    """One versioned system output used by the offline evaluator."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"]
    case_id: str = Field(min_length=1, max_length=200)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    no_answer: StrictBool
    unauthorized_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _normalize_nonblank(value, field_name="case_id")

    @field_validator(
        "retrieved_chunk_ids",
        "reranked_chunk_ids",
        "citation_chunk_ids",
        "unauthorized_chunk_ids",
    )
    @classmethod
    def validate_chunk_id_lists(
        cls,
        values: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        return _normalize_string_list(values, field_name=info.field_name)


class RagEvalMetrics(BaseModel):
    """Versioned aggregate metrics emitted by the offline evaluator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["rag-eval-metrics.v1"] = (
        RAG_EVAL_METRICS_SCHEMA_VERSION
    )
    evaluated_cases: int = Field(ge=1)
    retrieval_cases: int = Field(ge=0)
    reranker_cutoff: int = Field(
        gt=0,
        description="Number of reranked IDs treated as final answer context.",
    )
    recall_at_k: dict[str, float]
    reciprocal_rank_by_case: dict[str, float]
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    reranker_recall_at_cutoff: float = Field(ge=0.0, le=1.0)
    no_answer_precision: float | None = Field(ge=0.0, le=1.0)
    no_answer_recall: float | None = Field(ge=0.0, le=1.0)
    expected_no_answer_count: int = Field(
        ge=0,
        description="Number of cases labeled with expected_no_answer=true.",
    )
    predicted_no_answer_count: int = Field(
        ge=0,
        description="Number of predictions with no_answer=true.",
    )
    unauthorized_chunk_count: int = Field(
        ge=0,
        description=(
            "Unique (case_id, chunk_id) unauthorized retrieval events; repeated "
            "IDs within one case count once and the same ID in another case "
            "counts again."
        ),
    )
    unknown_citation_count: int = Field(
        ge=0,
        description=(
            "Unique (case_id, chunk_id) citation events absent from that case's "
            "final reranked context; repeated IDs within one case count once and "
            "the same ID in another case counts again."
        ),
    )


def _validate_cutoffs(ks: tuple[int, ...]) -> tuple[int, ...]:
    if not ks:
        raise ValueError("At least one retrieval cutoff is required.")
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("Every retrieval cutoff must be a positive integer.")
    if len(set(ks)) != len(ks):
        raise ValueError("Retrieval cutoff values must be unique.")
    return ks


def _validate_reranker_cutoff(reranker_cutoff: int) -> int:
    if (
        isinstance(reranker_cutoff, bool)
        or not isinstance(reranker_cutoff, int)
        or reranker_cutoff <= 0
    ):
        raise ValueError("Reranker cutoff must be a positive integer.")
    return reranker_cutoff


RecordT = TypeVar("RecordT", RagEvalCase, RagEvalPrediction)


def _index_unique(
    records: Sequence[RecordT],
    *,
    label: str,
) -> dict[str, RecordT]:
    indexed = {record.case_id: record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"{label} case IDs must be unique.")
    return indexed


def evaluate_predictions(
    cases: Iterable[RagEvalCase],
    predictions: Iterable[RagEvalPrediction],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    reranker_cutoff: int = 6,
) -> RagEvalMetrics:
    """Compute macro retrieval and no-answer metrics from exact case coverage.

    Retrieval cutoffs and reciprocal rank use the original ranked list, so every
    returned position counts even when an ID repeats. Set membership within each
    cutoff prevents repeated relevant IDs from inflating recall. Reranker recall
    and citation provenance use only the explicit final reranker cutoff.
    Unauthorized and unknown citation counts are unique ``(case_id, chunk_id)``
    events: duplicates within a case count once, while the same ID in a different
    case counts again. The evaluator intentionally does not infer answer
    correctness or citation entailment from retrieval labels.
    """

    cutoffs = _validate_cutoffs(ks)
    final_context_cutoff = _validate_reranker_cutoff(reranker_cutoff)
    case_list = list(cases)
    prediction_list = list(predictions)
    if not case_list:
        raise ValueError("At least one evaluation case is required.")

    cases_by_id = _index_unique(case_list, label="Evaluation")
    predictions_by_id = _index_unique(prediction_list, label="Prediction")
    case_ids = set(cases_by_id)
    prediction_ids = set(predictions_by_id)
    missing = sorted(case_ids - prediction_ids)
    extra = sorted(prediction_ids - case_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"Missing predictions for case IDs: {missing}.")
        if extra:
            details.append(f"Extra predictions for case IDs: {extra}.")
        raise ValueError(" ".join(details))

    retrieval_cases = [
        case
        for case in case_list
        if case.answerable and case.relevant_chunk_ids
    ]
    recall_totals = {k: 0.0 for k in cutoffs}
    reciprocal_rank_by_case: dict[str, float] = {}
    reranker_recall_total = 0.0

    for case in retrieval_cases:
        prediction = predictions_by_id[case.case_id]
        relevant = set(case.relevant_chunk_ids)
        retrieved_ids = prediction.retrieved_chunk_ids
        reranked_ids = set(
            prediction.reranked_chunk_ids[:final_context_cutoff]
        )

        for k in cutoffs:
            retrieved_at_k = set(retrieved_ids[:k])
            recall_totals[k] += len(relevant & retrieved_at_k) / len(relevant)

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant
            ),
            None,
        )
        reciprocal_rank_by_case[case.case_id] = (
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        )
        reranker_recall_total += len(relevant & reranked_ids) / len(relevant)

    retrieval_divisor = len(retrieval_cases)
    if retrieval_divisor:
        recall_at_k = {
            str(k): recall_totals[k] / retrieval_divisor for k in cutoffs
        }
        mean_reciprocal_rank = (
            sum(reciprocal_rank_by_case.values()) / retrieval_divisor
        )
        reranker_recall = reranker_recall_total / retrieval_divisor
    else:
        recall_at_k = {str(k): 0.0 for k in cutoffs}
        mean_reciprocal_rank = 0.0
        reranker_recall = 0.0

    true_positive = sum(
        predictions_by_id[case.case_id].no_answer and case.expected_no_answer
        for case in case_list
    )
    false_positive = sum(
        predictions_by_id[case.case_id].no_answer
        and not case.expected_no_answer
        for case in case_list
    )
    false_negative = sum(
        not predictions_by_id[case.case_id].no_answer
        and case.expected_no_answer
        for case in case_list
    )
    predicted_no_answer = true_positive + false_positive
    expected_no_answer = true_positive + false_negative

    unauthorized_chunk_count = sum(
        len(set(prediction.unauthorized_chunk_ids))
        for prediction in prediction_list
    )
    unknown_citation_count = sum(
        len(
            set(prediction.citation_chunk_ids)
            - set(
                prediction.reranked_chunk_ids[:final_context_cutoff]
            )
        )
        for prediction in prediction_list
    )

    return RagEvalMetrics(
        evaluated_cases=len(case_list),
        retrieval_cases=retrieval_divisor,
        reranker_cutoff=final_context_cutoff,
        recall_at_k=recall_at_k,
        reciprocal_rank_by_case=reciprocal_rank_by_case,
        mean_reciprocal_rank=mean_reciprocal_rank,
        reranker_recall_at_cutoff=reranker_recall,
        no_answer_precision=(
            true_positive / predicted_no_answer if predicted_no_answer else None
        ),
        no_answer_recall=(
            true_positive / expected_no_answer if expected_no_answer else None
        ),
        expected_no_answer_count=expected_no_answer,
        predicted_no_answer_count=predicted_no_answer,
        unauthorized_chunk_count=unauthorized_chunk_count,
        unknown_citation_count=unknown_citation_count,
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
