# No-Answer Policy

Runnable implementation: `model/agentic_rag/nodes.py` and `model/agentic_rag/graphs.py`.

## Principle

Document QA remains RAG-first and fail-closed for document/private facts. A question may become a grounded answer only after sufficient authorized evidence, exact citation validation, and accepted claim-evidence reflection. An eligible general-domain question may receive explicitly labeled model knowledge only after retrieval infrastructure successfully completes for at least one selected collection. Provider failure is never a reason to use model memory.

## Decision Table

| Condition | Result |
|---|---|
| Explicit active-session history intent | Citation-free conversational response |
| No completed accessible documents | Explicit document no-answer |
| Successful empty retrieval, eligible general question | Labeled general-model answer; no citations |
| Successful empty retrieval, document/private question | Explicit document no-answer |
| Complete retrieval failure | Generic backend service/provider error; rollback |
| OpenAI reranker error, refusal, timeout, or invalid output | Generic backend service/provider error; rollback |
| Grounding evaluator incomplete/unparsed HTTP-200 result | Retry once with independent bounded reasoning; if still unusable, generic backend service/provider error and rollback |
| `sufficient_evidence=false`, eligible general question | Labeled general-model answer; no citations |
| `sufficient_evidence=false`, document/private question | Explicit document no-answer |
| Valid reranked evidence, valid citations, accepted full coverage | Grounded answer with citations |
| Valid reranked evidence, accepted partial coverage | Cited document section plus labeled uncited general section |
| Citation validation failure | Explicit document no-answer |
| Semantic reflection `revise` or `no_answer` with valid citations | One grounded repair, then revalidation |
| Reflection provider/schema error | Generic backend service/provider error; rollback |
| Repair provider/schema error | Generic backend service/provider error; rollback |
| Rejected repair, eligible general question | Labeled general-model answer; no citations |
| Rejected repair, document/private question | Explicit document no-answer |
| Answer-generation provider outage | Generic backend service/provider error; not document no-answer |
| Conversation-generation provider outage | Generic backend service/provider error; not document no-answer |
| Pure general-generation provider outage | Generic backend service/provider error; rollback |
| Optional general supplement fails after accepted grounding | Return the accepted grounded answer only |

Every no-answer response has `no_answer=true` and `citations=[]`. An explicit
document no-answer additionally keeps the graph in grounded mode:

```text
no_answer=true
citations=[]
response_mode=grounded
```

The strict no-answer response says that the accessible uploaded documents do not contain enough evidence. A general-model answer instead begins with this server-owned label:

```text
> **Genel model bilgisi:** Aşağıdaki bölüm yüklediğiniz belgelerde doğrulanmamıştır.
```

Neither response exposes internal provider names, refusals, timeouts, validation details, or error messages.

## Provider-Outage Boundary

Evidence-path failure and response-provider outage are different terminal
conditions:

- A collection-isolated Qdrant failure may coexist with a safe answer only when
  another selected collection completed successfully. Complete retrieval,
  malformed retrieval payload, embedding, reranking, answer, repair, grounding,
  conversation, and pure-general provider failures produce a sanitized fatal
  `NodeError`; FastAPI raises `rag_provider_failure` and rolls back the turn.
- Successful empty or insufficient retrieval is not a provider failure. It may
  enter general fallback only when the deterministic policy confirms that the
  question does not require a private document, upload, file, page, person, or
  user-specific fact.

Neither response exposes internal provider details to the user.

## Evidence Gate

For the OpenAI provider, the raw Qdrant or application RRF score is not evidence sufficiency. Grounded generation requires all of the following:

1. A valid structured reranker result with `sufficient_evidence=true`.
2. At least one accepted candidate with `support=direct` at or above `RERANK_MIN_SCORE`.
3. Current-user and allowed-document checks before and after reranking.
4. An exact opaque evidence marker that maps unambiguously to retained server-owned chunk metadata.
5. Accepted structured claim-evidence reflection after citation validation.

`RERANKER_ALLOW_PARTIAL_SUPPORT=false` is the default. `RERANK_MIN_SCORE=0.50` is a local starting value, not a calibrated probability or a measured quality result. The `NO_ANSWER_MIN_SCORE` and `CITATION_MIN_SCORE` settings apply only to explicit `noop` or `heuristic` development providers.

## General-Knowledge and Conversation Boundaries

General generation receives the current question and bounded messages only from the active PostgreSQL chat session. It never receives retrieved chunks, source excerpts, document metadata, citation IDs, hidden provider errors, or cross-user content. The application strips citation-like markers and attaches no citation objects to a general-only response. Hybrid output preserves citations only for the accepted document section.

Conversation generation remains a separate branch that runs only when the user explicitly asks about that session's earlier messages. It produces no document citations and never treats history as document evidence.

No PostgreSQL migration, Qdrant collection recreation, or document re-ingestion is required for this policy.
