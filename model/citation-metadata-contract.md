# Citation Metadata Contract

Runnable implementation: `model/agentic_rag/schemas.py` and `model/agentic_rag/nodes.py`.

## Required Citation Fields

Each retrieved chunk must preserve:

- `user_id`
- `document_id`
- `document_name`
- `document_type`
- `page_number` or `slide_number`
- `chunk_id`
- `collection_name`
- `source_pipeline`
- `source_excerpt`
- `retrieval_score`
- `created_at`

## Display Fields

The final answer should expose:

- document name
- page or slide number
- chunk id
- short excerpt
- retrieval score
- collection name
- grounding indicator

## Validation Rule

If an answer sentence cannot be linked to at least one retrieved evidence chunk, the reflector must mark it as unsupported.

The current skeleton validates citation presence and score sufficiency. Sentence-level claim matching remains a future hardening step behind the same citation validation node.
