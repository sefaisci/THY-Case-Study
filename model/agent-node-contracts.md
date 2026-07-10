# Agent Node Contracts

Runnable implementation: `model/agentic_rag/nodes.py` and `model/agentic_rag/graphs.py`.

## Query Understanding Node

Input: `question`

Output: `normalized_question`, optional query hints.

Responsibility: Clarify intent and normalize the query without answering.

## Retrieval Planner Node

Input: `normalized_question`

Output: `retrieval_plan`

Responsibility: Choose `semantic_chunks`, `docling_fixed_chunks`, or both.

## Hybrid Retrieval Node

Input: `retrieval_plan`, `user_id`

Output: `retrieved_chunks`

Responsibility: Query Qdrant using dense and sparse retrieval with user-scoped payload filters.

Implementation note: the real Qdrant adapter currently exposes the dense query path and preserves the hybrid interface boundary. Sparse retrieval can be added behind the same adapter without changing graph nodes.

## Reranking Node

Input: `retrieved_chunks`, `question`

Output: `reranked_chunks`

Responsibility: Reorder evidence through a pluggable reranker. The notebook default is a no-op or lightweight heuristic reranker.

## Answer Generation Node

Input: `question`, `reranked_chunks`

Output: `draft_answer`

Responsibility: Generate an answer only from retrieved evidence.

## Citation Validation Node

Input: `draft_answer`, `reranked_chunks`

Output: `citations`, validation notes.

Responsibility: Check that claims are linked to source metadata.

## Reflection Node

Input: `draft_answer`, `citations`, `reranked_chunks`

Output: `reflection`

Responsibility: Identify unsupported claims, weak evidence, missing citations, or hallucination risk.

## Final Response Node

Input: `draft_answer`, `citations`, `reflection`

Output: `final_answer`

Responsibility: Return the grounded answer or no-answer response.

Implementation note: final response also returns a `RagResponse` object for notebook and future backend consumption.
