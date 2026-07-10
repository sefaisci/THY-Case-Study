"""LangGraph node implementations for the agentic RAG workflow."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .adapters import RagAdapters
from .schemas import (
    Citation,
    CitationValidationResult,
    CollectionName,
    NodeError,
    RagResponse,
    RagState,
    ReflectionResult,
    RetrievalPlan,
    RetrievedChunk,
)
from .settings import RagSettings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagNodeSet:
    """Factory object that exposes node callables bound to adapters."""

    adapters: RagAdapters
    settings: RagSettings

    def query_understanding(self, state: RagState) -> dict:
        """Normalize the user question and assign a coarse intent label."""

        question = state.get("question", "").strip()
        normalized = re.sub(r"\s+", " ", question)
        intent = _infer_query_intent(normalized)
        documents_available = self.settings.allowed_document_ids != ()
        retrieval_query = _build_retrieval_query(
            normalized,
            state.get("conversation_history", []),
        )
        if (
            documents_available
            and intent != "conversation_history"
            and self.adapters.query_rewriter is not None
        ):
            try:
                rewrite = self.adapters.query_rewriter.rewrite(retrieval_query)
                retrieval_query = _combine_query_variants(
                    rewrite.standalone_query,
                    rewrite.english_query,
                )
            except Exception:
                logger.warning(
                    "Retrieval query rewrite failed; using the deterministic fallback.",
                    extra={"event": "retrieval_query_rewrite_fallback"},
                    exc_info=True,
                )
        return {
            "normalized_question": normalized,
            "retrieval_query": retrieval_query,
            "query_intent": intent,
            "documents_available": documents_available,
        }

    def retrieval_planner(self, state: RagState) -> dict:
        """Choose one or both Qdrant collections for the current question."""

        question = state.get("normalized_question") or state.get("question", "")
        intent = state.get("query_intent", "general")
        collection_scope = state.get("collection_scope", "both")
        collections = _select_collections(question, intent, collection_scope)
        plan = RetrievalPlan(
            collections=collections,
            top_k=self.settings.retrieval_top_k,
            rerank_top_k=self.settings.rerank_top_k,
            dense_weight=self.settings.hybrid_dense_weight,
            sparse_weight=self.settings.hybrid_sparse_weight,
            reason=(
                f"Selected collections for explicit scope '{collection_scope}' "
                f"and intent '{intent}'."
            ),
        )
        return {
            "retrieval_plan": plan,
            "checked_collections": collections,
        }

    def hybrid_retrieval(self, state: RagState) -> dict:
        """Run user-scoped hybrid retrieval through the injected adapter."""

        try:
            user_id = state["user_id"]
            retrieval_query = (
                state.get("retrieval_query")
                or state.get("normalized_question")
                or state["question"]
            )
            plan = state["retrieval_plan"]
            dense_vector = self.adapters.embedding.embed_query(retrieval_query)
            chunks = self.adapters.retrieval.retrieve(
                query=retrieval_query,
                dense_vector=dense_vector,
                plan=plan,
                user_id=user_id,
            )
            return {"retrieved_chunks": chunks}
        except Exception as exc:  # pragma: no cover - exercised in connected runs
            return {
                "retrieved_chunks": [],
                "errors": [NodeError(node="hybrid_retrieval", message=str(exc))],
            }

    def reranking(self, state: RagState) -> dict:
        """Rerank retrieved chunks while preserving citation metadata."""

        try:
            plan = state["retrieval_plan"]
            question = state.get("normalized_question") or state["question"]
            chunks = state.get("retrieved_chunks", [])
            reranked = self.adapters.reranker.rerank(
                question=question,
                chunks=chunks,
                limit=plan.rerank_top_k,
            )
            return {
                "reranked_chunks": reranked,
                "evidence_sufficient": _has_sufficient_evidence(
                    reranked,
                    self.settings.no_answer_min_score,
                ),
            }
        except Exception as exc:  # pragma: no cover - exercised in connected runs
            retrieved = state.get("retrieved_chunks", [])
            return {
                "reranked_chunks": retrieved,
                "evidence_sufficient": _has_sufficient_evidence(
                    retrieved,
                    self.settings.no_answer_min_score,
                ),
                "errors": [NodeError(node="reranking", message=str(exc))],
            }

    def answer_generation(self, state: RagState) -> dict:
        """Generate an answer only from the retrieved evidence."""

        chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
        if not _has_sufficient_evidence(chunks, self.settings.no_answer_min_score):
            return {"draft_answer": "", "no_answer": True}

        system_prompt = (
            "You answer questions only from the retrieved document evidence. "
            "Do not use external knowledge. If the evidence is insufficient, "
            "say that the uploaded documents do not contain enough evidence. "
            "Use exact chunk citations in square brackets before sentence-ending "
            "punctuation, for example "
            "'The statement is supported [exact-chunk-id].' Never use numeric citation aliases. "
            "Attach citations to each paragraph or coherent group of related claims. A citation "
            "may support an immediately preceding multiline list or project tree when the cited "
            "chunk supports that entire group."
        )
        user_prompt = _build_grounded_answer_prompt(
            question=state.get("normalized_question") or state["question"],
            chunks=chunks[: self.settings.max_context_chunks],
            conversation_history=state.get("conversation_history", []),
        )
        try:
            answer = self.adapters.llm.complete(
                system_prompt,
                user_prompt,
                reasoning_effort=self.settings.self_service_reasoning_effort,
            )
            return {
                "draft_answer": answer,
                "no_answer": _is_explicit_no_answer(answer),
                "response_mode": "grounded",
            }
        except Exception as exc:  # pragma: no cover - exercised in connected runs
            return {
                "draft_answer": "",
                "no_answer": True,
                "errors": [NodeError(node="answer_generation", message=str(exc))],
            }

    def conversation_generation(self, state: RagState) -> dict:
        """Generate a citation-free answer from only this chat session's context."""

        system_prompt = (
            "You are a helpful conversational assistant. Respond naturally in the same "
            "language as the user's current message. No sufficiently relevant document "
            "evidence is available for this turn, or the user is asking about the chat "
            "itself. Do not claim that the supplied session history is document evidence. "
            "Do not invent document facts, sources, citations, footnotes, or bracketed "
            "citation markers. Use only the supplied session history as short-term memory, "
            "and never imply access to another chat session. If the user asks what they "
            "previously wrote, said, or asked, summarize only prior user-role messages in "
            "the supplied history; do not include the current question. If no prior user "
            "message exists, say so clearly."
        )
        user_prompt = _build_conversational_prompt(
            question=state.get("normalized_question") or state["question"],
            conversation_history=state.get("conversation_history", []),
        )
        try:
            answer = self.adapters.llm.complete(
                system_prompt,
                user_prompt,
                reasoning_effort=self.settings.self_service_reasoning_effort,
            ).strip()
            if not answer:
                raise RuntimeError("The conversational model returned an empty response.")
            return {
                "draft_answer": answer,
                "response_mode": "conversational",
                "citations": [],
                "no_answer": False,
            }
        except Exception as exc:  # pragma: no cover - connected provider failure
            return {
                "draft_answer": "",
                "response_mode": "conversational",
                "citations": [],
                "no_answer": True,
                "errors": [
                    NodeError(node="conversation_generation", message=str(exc))
                ],
            }

    def citation_validation(self, state: RagState) -> dict:
        """Validate exact chunk markers and attach only explicitly cited evidence."""

        draft_answer = state.get("draft_answer", "").strip()
        user_id = state.get("user_id", "")
        if state.get("no_answer", False) or _is_explicit_no_answer(draft_answer):
            validation = CitationValidationResult(
                is_valid=False,
                missing_citation_sentences=["The answer explicitly reports insufficient document evidence."],
            )
            return {
                "citation_validation": validation,
                "citations": [],
                "no_answer": True,
            }
        chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        markers = _extract_citation_markers(draft_answer)
        cited_chunk_ids = list(dict.fromkeys(markers))
        unknown = [chunk_id for chunk_id in cited_chunk_ids if chunk_id not in chunk_by_id]
        cross_user = [
            chunk_id
            for chunk_id in cited_chunk_ids
            if chunk_id in chunk_by_id and chunk_by_id[chunk_id].user_id != user_id
        ]
        weak = [
            chunk_id
            for chunk_id in cited_chunk_ids
            if chunk_id in chunk_by_id
            and chunk_by_id[chunk_id].effective_score < self.settings.citation_min_score
        ]
        unknown = list(dict.fromkeys([*unknown, *weak]))
        missing_sentences = [
            sentence[:500]
            for sentence in _answer_sentences(draft_answer)
            if not _extract_citation_markers(sentence)
        ]
        valid_ids = [
            chunk_id
            for chunk_id in cited_chunk_ids
            if chunk_id in chunk_by_id
            and chunk_id not in cross_user
            and chunk_id not in unknown
        ]
        validation = CitationValidationResult(
            is_valid=bool(draft_answer and valid_ids)
            and not unknown
            and not cross_user,
            cited_chunk_ids=valid_ids,
            unknown_chunk_ids=unknown,
            cross_user_chunk_ids=cross_user,
            missing_citation_sentences=missing_sentences,
        )
        citations = [
            _chunk_to_citation(chunk_by_id[chunk_id], self.settings.citation_min_score)
            for chunk_id in valid_ids
        ]
        return {
            "citation_validation": validation,
            "citations": citations,
            "no_answer": not validation.is_valid,
        }

    def claim_evidence_reflection(self, state: RagState) -> dict:
        """Run structured LLM claim-evidence evaluation after deterministic validation."""

        validation = state.get("citation_validation")
        draft_answer = state.get("draft_answer", "").strip()
        if validation is None or not validation.is_valid:
            missing = []
            if validation is None:
                missing = ["Citation validation did not run."]
            else:
                missing = validation.missing_citation_sentences
            reflection = ReflectionResult(
                is_grounded=False,
                hallucination_risk="high",
                decision="no_answer",
                unsupported_claims=(validation.unknown_chunk_ids if validation else []),
                missing_citations=missing,
                notes="Deterministic citation validation failed.",
            )
            return {"reflection": reflection, "no_answer": True}

        chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        cited_chunks = [chunk_by_id[chunk_id] for chunk_id in validation.cited_chunk_ids]
        try:
            reflection = self.adapters.grounding.evaluate(
                question=state.get("normalized_question") or state["question"],
                draft_answer=draft_answer,
                cited_chunks=cited_chunks,
            )
            allowed_ids = set(validation.cited_chunk_ids)
            evaluator_ids = {
                chunk_id
                for claim in reflection.claims
                for chunk_id in claim.cited_chunk_ids
            }
            invalid_evaluator_ids = sorted(evaluator_ids - allowed_ids)
            accepted = (
                reflection.is_grounded
                and reflection.decision == "accept"
                and reflection.hallucination_risk != "high"
                and not reflection.unsupported_claims
                and not reflection.missing_citations
                and not invalid_evaluator_ids
            )
            if not accepted:
                reflection = reflection.model_copy(
                    update={
                        "is_grounded": False,
                        "decision": "no_answer",
                        "notes": (
                            f"{reflection.notes} Invalid evaluator citation IDs: {invalid_evaluator_ids}."
                            if invalid_evaluator_ids
                            else reflection.notes
                        ),
                    }
                )
            return {"reflection": reflection, "no_answer": not accepted}
        except Exception as exc:  # pragma: no cover - connected provider failure
            reflection = ReflectionResult(
                is_grounded=False,
                hallucination_risk="high",
                decision="no_answer",
                unsupported_claims=[],
                missing_citations=[],
                notes="Grounding evaluator failed; safe no-answer policy applied.",
            )
            return {
                "reflection": reflection,
                "no_answer": True,
                "errors": [NodeError(node="claim_evidence_reflection", message=str(exc))],
            }

    def reflection(self, state: RagState) -> dict:
        """Backward-compatible alias for the claim-evidence reflection node."""

        return self.claim_evidence_reflection(state)

    def final_response(self, state: RagState) -> dict:
        """Return a grounded final answer or an explicit no-answer response."""

        if state.get("response_mode") == "conversational":
            answer = state.get("draft_answer", "").strip()
            no_answer = not bool(answer)
            if no_answer:
                answer = _build_no_answer_message(state)
            response = RagResponse(
                answer=answer,
                citations=[],
                no_answer=no_answer,
                checked_collections=state.get("checked_collections", []),
                citation_validation=None,
                reflection=None,
                errors=state.get("errors", []),
            )
            return {
                "final_answer": answer,
                "response": response,
                "citations": [],
                "no_answer": no_answer,
            }

        chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
        reflection = state.get("reflection")
        citation_validation = state.get("citation_validation")
        should_no_answer = state.get("no_answer", False) or not _has_sufficient_evidence(
            chunks,
            self.settings.no_answer_min_score,
        )
        if reflection is not None and not reflection.is_grounded:
            should_no_answer = True
        if citation_validation is None or not citation_validation.is_valid:
            should_no_answer = True

        if should_no_answer:
            answer = _build_no_answer_message(state)
            citations: list[Citation] = []
            no_answer = True
        else:
            citations = state.get("citations", [])
            answer = _replace_internal_citation_markers(
                state.get("draft_answer", "").strip(),
                citations,
            )
            no_answer = False

        response = RagResponse(
            answer=answer,
            citations=citations,
            no_answer=no_answer,
            checked_collections=state.get("checked_collections", []),
            citation_validation=citation_validation,
            reflection=reflection,
            errors=state.get("errors", []),
        )
        return {
            "final_answer": answer,
            "response": response,
            "citations": citations,
            "no_answer": no_answer,
        }


def route_after_query_understanding(state: RagState) -> str:
    """Bypass document retrieval for explicit session-history questions."""

    if (
        state.get("query_intent") == "conversation_history"
        or not state.get("documents_available", True)
    ):
        return "conversation_generation"
    return "retrieval_subgraph"


def route_after_retrieval(state: RagState) -> str:
    """Route sufficient evidence to grounding and all other turns to normal chat."""

    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
    if state.get("evidence_sufficient", bool(chunks)):
        return "answer_subgraph"
    return "conversation_generation"


def route_after_answer(state: RagState) -> str:
    """Keep accepted grounded answers; conversationally recover rejected evidence."""

    validation = state.get("citation_validation")
    reflection = state.get("reflection")
    accepted = (
        not state.get("no_answer", False)
        and validation is not None
        and validation.is_valid
        and reflection is not None
        and reflection.is_grounded
    )
    return "final_response" if accepted else "conversation_generation"


def _infer_query_intent(question: str) -> str:
    if _is_conversation_history_question(question):
        return "conversation_history"
    lowered = question.lower()
    if any(term in lowered for term in ["table", "image", "diagram", "slide", "page"]):
        return "visual_or_layout"
    if any(term in lowered for term in ["exact", "definition", "policy", "rule"]):
        return "textual_lookup"
    if any(term in lowered for term in ["compare", "summarize", "overview"]):
        return "synthesis"
    return "general"


def _select_collections(
    question: str,
    intent: str,
    collection_scope: str = "both",
) -> list[CollectionName]:
    if collection_scope == "semantic":
        return ["semantic_chunks"]
    if collection_scope == "docling":
        return ["docling_fixed_chunks"]
    if collection_scope == "both":
        return ["semantic_chunks", "docling_fixed_chunks"]
    lowered = question.lower()
    if intent == "visual_or_layout":
        return ["semantic_chunks"]
    if intent == "textual_lookup":
        return ["docling_fixed_chunks"]
    if "semantic" in lowered and "docling" not in lowered:
        return ["semantic_chunks"]
    if "docling" in lowered and "semantic" not in lowered:
        return ["docling_fixed_chunks"]
    return ["semantic_chunks", "docling_fixed_chunks"]


def _has_sufficient_evidence(chunks: list[RetrievedChunk], min_score: float) -> bool:
    return bool(chunks) and max(chunk.effective_score for chunk in chunks) >= min_score


def _build_grounded_answer_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    conversation_history: list[object] | None = None,
) -> str:
    evidence_lines = []
    for chunk in chunks:
        evidence_lines.append(
            f"[{chunk.chunk_id}]\n"
            f"Text: {chunk.text[:4000]}\n"
            f"Source excerpt: {chunk.source_excerpt}\n"
            f"Document: {chunk.document_name}; Location: {chunk.display_location}; "
            f"Chunk: {chunk.chunk_id}; Collection: {chunk.collection_name}; "
            f"Score: {chunk.effective_score:.3f}"
        )
    evidence = "\n\n".join(evidence_lines)
    history_lines = []
    for turn in (conversation_history or [])[-12:]:
        role = str(getattr(turn, "role", ""))
        content = str(getattr(turn, "content", ""))
        if role in {"user", "assistant"} and content.strip():
            history_lines.append(f"{role.title()}: {content[:2000]}")
    history = "\n".join(history_lines) or "No prior turns in this chat session."
    return (
        "Session history may clarify references in the current question, but it is not "
        "document evidence and must never be cited as a source.\n"
        f"Session history:\n{history}\n\n"
        f"Current question:\n{question}\n\n"
        f"Retrieved evidence:\n{evidence}\n\n"
        "Write a concise answer grounded only in the evidence. Cite every factual sentence "
        "or coherent group of claims with one or more exact [chunk_id] markers shown above. "
        "For a multiline list or project tree supported by one chunk, one marker may follow the "
        "whole group. The grounding evaluator will check the full draft against cited chunks."
    )


def _build_conversational_prompt(
    question: str,
    conversation_history: list[object] | None = None,
) -> str:
    """Serialize bounded, role-preserving short-term context for normal chat."""

    history = []
    for turn in (conversation_history or [])[-20:]:
        role = str(getattr(turn, "role", ""))
        content = str(getattr(turn, "content", "")).strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content[:4_000]})
    return json.dumps(
        {
            "session_history": history,
            "current_user_message": question,
        },
        ensure_ascii=False,
    )


def _extract_citation_markers(text: str) -> list[str]:
    return re.findall(r"\[([a-zA-Z0-9_.:-]+)\]", text)


def _answer_sentences(answer: str) -> list[str]:
    if not answer.strip():
        return []
    normalized = re.sub(
        r"([.!?])\s+((?:\[[a-zA-Z0-9_.:-]+\]\s*)+)",
        r" \2\1",
        answer.strip(),
    )
    segments = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [segment.strip() for segment in segments if re.search(r"[A-Za-zÀ-ž0-9]", segment)]


def _is_explicit_no_answer(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer.strip().casefold())
    indicators = (
        "uploaded documents do not contain enough evidence",
        "documents do not contain enough evidence",
        "belgeler yeterli kanıt içermiyor",
        "belgelerde yeterli bilgi bulunmuyor",
        "dokümanlarda yeterli bilgi bulunmuyor",
    )
    patterns = (
        r"(?:uploaded )?documents?.{0,120}(?:do not|don't).{0,60}(?:enough|sufficient) (?:evidence|information)",
        r"(?:yüklenen )?(?:dokümanlar|dokumanlar|belgeler).{0,120}yeterli (?:kanıt|kanit|bilgi).{0,60}(?:içermiyor|icermiyor|bulunmuyor)",
        r"(?:dokümanlarda|dokumanlarda|belgelerde).{0,120}yeterli (?:kanıt|kanit|bilgi).{0,60}(?:yok|bulunmuyor)",
    )
    return any(indicator in normalized for indicator in indicators) or any(
        re.search(pattern, normalized) for pattern in patterns
    )


def _chunk_to_citation(chunk: RetrievedChunk, citation_min_score: float) -> Citation:
    grounding = "grounded" if chunk.effective_score >= citation_min_score else "weak"
    return Citation(
        document_name=chunk.document_name,
        document_id=chunk.document_id,
        page_number=chunk.page_number,
        slide_number=chunk.slide_number,
        chunk_id=chunk.chunk_id,
        source_excerpt=chunk.source_excerpt,
        retrieval_score=chunk.effective_score,
        collection_name=chunk.collection_name,
        ingestion_method=(
            "semantic" if chunk.collection_name == "semantic_chunks" else "docling"
        ),
        source_pipeline=chunk.source_pipeline,
        grounding_indicator=grounding,
    )


def _build_retrieval_query(question: str, conversation_history: list[object]) -> str:
    previous_user_turns = [
        str(getattr(turn, "content", "")).strip()
        for turn in conversation_history[-8:]
        if getattr(turn, "role", None) == "user"
        and str(getattr(turn, "content", "")).strip()
    ]
    if not previous_user_turns or not _is_context_dependent_question(question):
        return question
    return f"{previous_user_turns[-1]}\nFollow-up question: {question}"


def _is_context_dependent_question(question: str) -> bool:
    """Return whether retrieval needs the immediately preceding user turn."""

    normalized = re.sub(r"\s+", " ", question.strip().casefold())
    if not normalized:
        return False
    prefixes = (
        "and ",
        "how about",
        "peki",
        "then ",
        "what about",
        "ya ",
    )
    if normalized.startswith(prefixes):
        return True
    terms = set(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))
    references = {
        "buna",
        "bunda",
        "bunlar",
        "bunun",
        "it",
        "its",
        "onlar",
        "onda",
        "onun",
        "that",
        "them",
        "these",
        "they",
        "this",
        "those",
    }
    return len(terms) <= 24 and bool(terms & references)


def _is_conversation_history_question(question: str) -> bool:
    """Recognize explicit requests about prior messages in the active chat."""

    normalized = re.sub(r"\s+", " ", question.strip().casefold())
    if not normalized:
        return False
    patterns = (
        r"\b(?:daha önce|daha once|önceden|onceden).{0,40}\b(?:ne(?:ler)?|hangi).{0,30}\b(?:yazd|söyled|soyled|sord|konuşt|konust)",
        r"\b(?:ben )?(?:sana )?ne(?:ler)? (?:yazd|söyled|soyled|sord)",
        r"\b(?:önceki|onceki|geçmiş|gecmis).{0,25}\b(?:mesaj|soru|sohbet|konuşma|konusma)",
        r"\b(?:sohbet|konuşma|konusma).{0,20}\bgeçmiş",
        r"\bwhat (?:did|have) i.{0,20}\b(?:say|write|ask).{0,30}\b(?:before|earlier|previously)",
        r"\bwhat (?:have )?i (?:said|written|asked) (?:before|earlier|previously)",
        r"\b(?:previous|earlier|prior).{0,20}\b(?:message|question|conversation|chat)",
        r"\b(?:conversation|chat) history\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _combine_query_variants(*queries: str) -> str:
    """Join distinct query-language variants without duplicating content."""

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = re.sub(r"\s+", " ", query.strip())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return "\n".join(unique)


def _replace_internal_citation_markers(
    answer: str,
    citations: list[Citation],
) -> str:
    """Replace internal chunk IDs with stable, user-facing citation numbers."""

    number_by_chunk = {
        citation.chunk_id: index
        for index, citation in enumerate(citations, start=1)
    }

    def replace(match: re.Match[str]) -> str:
        chunk_id = match.group(1)
        number = number_by_chunk.get(chunk_id)
        return f"[{number}]" if number is not None else match.group(0)

    return re.sub(r"\[([a-zA-Z0-9_.:-]+)\]", replace, answer)


def _build_no_answer_message(state: RagState) -> str:
    collections = state.get("checked_collections", [])
    checked = ", ".join(collections) if collections else "the configured document collections"
    return (
        "The uploaded documents do not contain enough evidence to answer this "
        f"question. Checked collections: {checked}. Please upload more relevant "
        "documents or rephrase the question."
    )
