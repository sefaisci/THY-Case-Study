# LangGraph State Contract

## Purpose

Define the shared state shape that notebook experiments will use before the logic is moved into production modules.

Runnable implementation: `model/agentic_rag/schemas.py`.

## State Fields

- `user_id`: Current user identifier used for metadata filtering.
- `question`: Original user question.
- `normalized_question`: Query-understanding output.
- `retrieval_plan`: Collection selection and retrieval parameters.
- `retrieved_chunks`: Raw chunks returned from Qdrant.
- `reranked_chunks`: Reranked evidence candidates.
- `draft_answer`: Answer generated from retrieved evidence.
- `citations`: Citation records attached to the answer.
- `reflection`: Reflector findings about grounding, citation quality, and hallucination risk.
- `final_answer`: Final answer or no-answer response.
- `errors`: Recoverable workflow errors.
- `response`: External `RagResponse` returned by the notebook runner.

## Invariants

- `user_id` must exist before retrieval.
- `retrieved_chunks` must only contain chunks that pass the user-scoped Qdrant payload filter.
- `final_answer` must not contain unsupported claims.
- If evidence is insufficient, `final_answer` must be a clear no-answer response.
- `errors` are append-only across graph nodes.
