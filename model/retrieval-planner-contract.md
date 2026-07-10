# Retrieval Planner Contract

## Collection Choices

- `semantic_chunks`: Use for visually structured pages, tables, page-level semantics, and chunking produced by the LLM image pipeline.
- `docling_fixed_chunks`: Use for direct text questions, factual lookup, and chunks produced from Docling extraction.
- `both`: Use when the question could benefit from semantic and textual evidence.

## Retrieval Defaults

- `RETRIEVAL_TOP_K=12`
- `RERANK_TOP_K=6`
- `HYBRID_DENSE_WEIGHT=0.65`
- `HYBRID_SPARSE_WEIGHT=0.35`

Runtime implementation: `model/agentic_rag/adapters.py` and `model/agentic_rag/nodes.py`.

## Required Payload Filter

Every Qdrant query must include:

```text
user_id == current_user_id
```

Future production filters may add document-level ACL or team-level scopes.

The fake notebook retriever also includes a cross-user chunk to smoke-check that user-scoped filtering prevents leakage.

## Output Shape

```text
retrieval_plan = {
  collections: ["semantic_chunks" | "docling_fixed_chunks"],
  top_k: int,
  rerank_top_k: int,
  dense_weight: float,
  sparse_weight: float,
  reason: str
}
```
