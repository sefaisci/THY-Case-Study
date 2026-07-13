# Agent Node Contracts

Runnable implementation: `model/agentic_rag/nodes.py` and `model/agentic_rag/graphs.py`.

## Query Understanding Node

Input: `question`

Output: `normalized_question`, one verbatim query variant, and at most one
distinct standalone variant for a referential follow-up.

Responsibility: Clarify intent and normalize the query without answering.

## Retrieval Planner Node

Input: `normalized_question`

Output: `retrieval_plan`

Responsibility: Choose `semantic_chunks`, `docling_fixed_chunks`, or both.

## Hybrid Retrieval Node

Input: `retrieval_plan`, `user_id`

Output: `retrieved_chunks`

Responsibility: Query Qdrant using dense and sparse retrieval with user-scoped payload filters.

Implementation note: every selected collection is queried with owner-scoped
dense and configured sparse prefetches. Qdrant performs weighted RRF within the
collection, then the application forms a collection-balanced candidate union.

## Reranking Node

Input: `retrieved_chunks`, `question`

Output: `reranked_chunks`, `evidence_sufficient`, and typed node errors.

Responsibility: Use one structured OpenAI relevance judgment over authorized
candidates, accept configured direct support above the local threshold, and
fail closed to explicit document no-answer on provider or validation failure.
No-op and heuristic providers remain explicit offline-development options.

## Answer Generation Node

Input: `question`, `reranked_chunks`

Output: `draft_answer`

Responsibility: Generate an answer only from retrieved evidence.

## Citation Validation Node

Input: `draft_answer`, `reranked_chunks`

Output: `citations`, validation notes.

Responsibility: Require exact collision-safe evidence markers and map them only
to retained server-owned source metadata. Unknown, ambiguous, weak, and
cross-user markers are hard failures.

## Reflection Node

Input: `draft_answer`, `citations`, `reranked_chunks`

Output: `reflection`

Responsibility: Identify unsupported claims, weak evidence, missing citations, or hallucination risk.

## Final Response Node

Input: `draft_answer`, `citations`, `reflection`

Output: `final_answer`

Responsibility: Return a grounded answer only after accepted citation validation
and reflection; otherwise return `no_answer=true` with `citations=[]`.

Implementation note: final response also returns a `RagResponse` object for notebook and future backend consumption.
