"""Deterministic policy and composition helpers for general-model fallback."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel


GENERAL_KNOWLEDGE_NOTICE = (
    "> **Genel model bilgisi:** Aşağıdaki bölüm yüklediğiniz "
    "belgelerde doğrulanmamıştır."
)


class FallbackDecision(BaseModel):
    """Deterministic eligibility decision for model-knowledge fallback."""

    eligible: bool
    reason: str


_DOCUMENT_REFERENCE = re.compile(
    r"\b(?:uploaded|upload|document|documents|file|files|pdf|docx|pptx|cv|"
    r"page|pages|slide|slides|chunk|chunks|yükle\w*|belge\w*|doküman\w*|"
    r"dokuman\w*|dosya\w*|sayfa\w*|slayt\w*|sunum\w*|sözleşme\w*|"
    r"sozlesme\w*)\b",
    re.IGNORECASE,
)
_PRIVATE_ATTRIBUTE = re.compile(
    r"\b(?:phone|telephone|email|e-mail|address|salary|identity|passport|"
    r"employee|personnel|ssn|telephone number|telefon\w*|e-posta\w*|"
    r"eposta\w*|adres\w*|maaş\w*|maas\w*|kimlik\w*|pasaport\w*|"
    r"çalışan\w*|calisan\w*|personel\w*|tc kimlik\w*)\b",
    re.IGNORECASE,
)
_PERSON_LOOKUP = re.compile(
    r"\b(?:who is|kimdir|kim olduğu|kim oldugu|where does .* live|nerede yaşar|"
    r"nerede yasar)\b",
    re.IGNORECASE,
)
_FILE_SCOPED_SUBJECT = re.compile(
    r"\b(?:project structure|repository structure|codebase structure|"
    r"(?:technical )?(?:assessment|evaluation) (?:content|details|summary)|"
    r"proje yapıs\w*|proje yapis\w*|repo yapıs\w*|repo yapis\w*|"
    r"kod tabanı yapıs\w*|kod tabani yapis\w*|"
    r"(?:teknik )?değerlendirme içeri\w*|"
    r"(?:teknik )?degerlendirme icer\w*)\b",
    re.IGNORECASE,
)
_INTERNAL_CITATION = re.compile(
    r"\[\[chunk:[^\]]+\]\]|\[[A-Za-z0-9_.:-]{2,200}\]"
)
_RENDERED_CITATION = re.compile(r"(?<!\w)\[\d+(?:\s*,\s*\d+)*\]")


def assess_general_fallback_eligibility(
    question: str,
    *,
    retrieval_succeeded: bool,
) -> FallbackDecision:
    """Allow fallback only for general knowledge after successful retrieval."""

    if not retrieval_succeeded:
        return FallbackDecision(eligible=False, reason="retrieval_failed")
    normalized = _normalize(question)
    if not normalized:
        return FallbackDecision(eligible=False, reason="empty_question")
    if _DOCUMENT_REFERENCE.search(normalized):
        return FallbackDecision(eligible=False, reason="document_specific")
    if _PRIVATE_ATTRIBUTE.search(normalized):
        return FallbackDecision(eligible=False, reason="private_attribute")
    if _PERSON_LOOKUP.search(normalized):
        return FallbackDecision(eligible=False, reason="person_specific")
    if _FILE_SCOPED_SUBJECT.search(normalized):
        return FallbackDecision(eligible=False, reason="file_scoped_subject")
    return FallbackDecision(eligible=True, reason="general_knowledge")


def strip_document_citation_markers(text: str) -> str:
    """Remove application citation syntax from ungrounded model text."""

    without_internal = _INTERNAL_CITATION.sub("", text)
    without_rendered = _RENDERED_CITATION.sub("", without_internal)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in without_rendered.splitlines()]
    return "\n".join(lines).strip()


def compose_general_knowledge_answer(text: str) -> str:
    """Prepend the server-owned warning to citation-free model knowledge."""

    sanitized = strip_document_citation_markers(text)
    return f"{GENERAL_KNOWLEDGE_NOTICE}\n\n{sanitized}".strip()


def compose_hybrid_answer(grounded: str, general: str) -> str:
    """Place cited document facts before the labeled general supplement."""

    sanitized = strip_document_citation_markers(general)
    return (
        "## Belgelerden bulunanlar\n\n"
        f"{grounded.strip()}\n\n"
        "## Genel model bilgisi\n\n"
        f"{GENERAL_KNOWLEDGE_NOTICE}\n\n"
        f"{sanitized}"
    ).strip()


def _normalize(text: str) -> str:
    """Return stable Unicode text for conservative lexical policy checks."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
