"""Command-line runner for deterministic RAG evaluation JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .rag_metrics import RagEvalCase, RagEvalPrediction, evaluate_predictions

ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonlInputError(ValueError):
    """A location-aware error raised for an invalid JSONL input record."""


def _format_validation_error(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "record"
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def load_jsonl(path: str | Path, model_type: type[ModelT]) -> list[ModelT]:
    """Load strict one-object-per-line JSONL with path and line diagnostics."""

    input_path = Path(path)
    try:
        raw_bytes = input_path.read_bytes()
    except OSError as error:
        raise JsonlInputError(f"{input_path}: unable to read input: {error}") from error

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        line_number = raw_bytes[: error.start].count(b"\n") + 1
        raise JsonlInputError(
            f"{input_path}:{line_number}: input is not valid UTF-8."
        ) from error

    records: list[ModelT] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        location = f"{input_path}:{line_number}"
        if not line:
            raise JsonlInputError(
                f"{location}: blank JSONL records are not allowed."
            )
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as error:
            raise JsonlInputError(
                f"{location}: malformed JSON: {error.msg}."
            ) from error
        if not isinstance(raw_record, dict):
            raise JsonlInputError(
                f"{location}: JSONL record must be an object."
            )
        try:
            records.append(model_type.model_validate(raw_record))
        except ValidationError as error:
            details = _format_validation_error(error)
            raise JsonlInputError(
                f"{location}: record failed {model_type.__name__} validation: "
                f"{details}"
            ) from error
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate versioned RAG retrieval predictions offline.",
    )
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        metavar="K",
        help="Positive, unique retrieval cutoffs (default: 5 10 20).",
    )
    parser.add_argument(
        "--reranker-cutoff",
        type=int,
        default=6,
        metavar="K",
        help="Final reranked context cutoff (default: 6).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load inputs, evaluate them, and emit one JSON metrics document."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        cases = load_jsonl(arguments.cases, RagEvalCase)
        predictions = load_jsonl(arguments.predictions, RagEvalPrediction)
        metrics = evaluate_predictions(
            cases,
            predictions,
            ks=tuple(arguments.ks),
            reranker_cutoff=arguments.reranker_cutoff,
        )
    except (JsonlInputError, ValueError) as error:
        parser.error(str(error))
    print(metrics.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
