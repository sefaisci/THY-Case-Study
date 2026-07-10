# No-Answer Policy

Runnable implementation: `model/agentic_rag/nodes.py`.

## Principle

Insufficient document evidence must never be presented as a grounded answer. It also must not end an otherwise valid conversation.

## Grounded Mode

Use grounded mode only when retrieval meets `NO_ANSWER_MIN_SCORE`, the draft contains at least one exact owner-scoped chunk ID above `CITATION_MIN_SCORE`, and structured claim-evidence evaluation accepts the complete draft. Unknown, cross-user, and weak chunk IDs remain hard failures. A single exact marker may cover an adjacent coherent multiline list or project tree, but the cited chunk must support every factual item in that group.

## Conversational Fallback

Use citation-free conversational mode when:

- the user has no completed documents;
- retrieval returns no chunks;
- every chunk remains below the evidence threshold;
- the grounded model explicitly declines for insufficient evidence;
- citation or structured grounding rejects the grounded draft; or
- the user asks about earlier messages in the active chat session.

Conversational generation receives only bounded history from that PostgreSQL chat session. It must not invent document facts, emit document citations, treat session history as document evidence, or access another session. Explicit history questions summarize prior user messages and exclude the current question.

Return a terminal no-answer or provider error only when both the grounded path and conversational recovery fail to produce a usable response.
