# LangGraph State Contract

The executable source of truth is `model/agentic_rag/schemas.py::RagState`.
`notebook/08_langgraph_rag_flow_sandbox.ipynb` groups and validates every field;
an unclassified new state field fails notebook execution.

## Request and Conversation

- `user_id`, `thread_id`, `collection_scope`
- `conversation_history`, `question`

## Query Understanding

- `normalized_question`, `retrieval_query`, `retrieval_turn_sequence`
- `query_variants`, `query_variant`, `query_intent`
- `documents_available`

## Retrieval

- `retrieval_plan`, `variant_results`
- `retrieved_chunks`, `reranked_chunks`
- `retrieval_succeeded`
- `successful_retrieval_collections`, `failed_retrieval_collections`
- `evidence_sufficient`, `checked_collections`

## Grounded Answer

- `draft_answer`, `grounded_draft`
- `grounded_repair_attempted`, `grounded_repair_feedback`
- `grounded_answer`
- `citations`, `citation_validation`, `reflection`

## Fallback and Terminal

- `general_knowledge_answer`
- `fallback_eligible`, `fallback_reason`
- `response_mode`
- `final_answer`, `response`, `no_answer`
- `errors`

## Reducers and Turn Reset

- `variant_results` uses an associative, commutative, and idempotent reducer so
  parallel `Send` completion order cannot change the retained result identity.
- `errors` uses append semantics for typed node failures.
- `query_understanding` resets turn-local retrieval, generation, citation,
  fallback, reflection, and terminal fields through explicit values and
  LangGraph `Overwrite` where reducer state must be replaced.

## Invariants

- `user_id` must exist before any retrieval operation.
- Retrieved chunks must pass server-side owner/document filtering and defensive
  payload validation before reranking.
- A grounded final response requires sufficient evidence, valid citations, and
  accepted reflection.
- A general-only response has no document citations and carries the fixed
  server-owned disclosure.
- An explicit no-answer has `no_answer=true` and `citations=[]`.
- Fatal provider, payload, ACL, or complete retrieval errors cannot route to
  general-knowledge generation.
