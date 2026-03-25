"""Project Aegis — Intent classifier.

Sprint 1.2 Task 9: Classifies user messages into intent types with confidence
scoring and complexity tier assignment for model routing (ID-01).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class IntentType(str, Enum):
    """Supported intent categories."""

    DIRECT_CHAT = "direct_chat"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TOOL_USE = "tool_use"
    COMPLEX_DELEGATION = "complex_delegation"
    MULTI_STEP = "multi_step"


class ComplexityTier(str, Enum):
    """Complexity tiers that map to provider routing (DR-010)."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    HEAVY = "heavy"


class OutputLength(str, Enum):
    """Expected output length categories."""

    SHORT = "short"  # <200 tokens
    MEDIUM = "medium"  # 200-1000 tokens
    LONG = "long"  # >1000 tokens


@dataclass(frozen=True)
class ClassificationResult:
    """Result of intent classification."""

    intent: IntentType
    confidence: float
    complexity: ComplexityTier
    expected_output_length: OutputLength
    uses_graph: bool  # True = full LangGraph, False = fast-path

    @property
    def is_simple(self) -> bool:
        """Whether this should use the fast-path chain."""
        return not self.uses_graph


# --- Keyword / pattern sets for heuristic classification ---

_TOOL_KEYWORDS = frozenset({
    "search", "google", "look up", "lookup", "find information",
    "search the web", "search for", "find me", "check online",
    "search online", "web search",
})

_KNOWLEDGE_KEYWORDS = frozenset({
    "what did we", "remind me", "what were", "show me what",
    "what notes", "find my notes", "our last conversation",
    "what i saved", "we discussed", "we talked about",
    "my notes on", "we decided",
})

_COMPLEX_KEYWORDS = frozenset({
    "research", "analyze", "compare and", "investigate",
    "find all", "identify", "assess", "evaluate",
    "review my", "check for", "audit",
    "root cause", "suggest a fix", "propose fix",
})

_MULTI_STEP_MARKERS = frozenset({
    "then", "after that", "next", "first.*then",
    "and then", "followed by", "step 1", "step 2",
})

_LONG_FORM_KEYWORDS = frozenset({
    "write a comprehensive", "write a detailed", "write a complete",
    "write a full", "write a thorough", "draft a detailed",
    "draft a complete", "create a detailed", "create a comprehensive",
    "guide to", "documentation for", "runbook", "strategy document",
    "retrospective", "specification for",
})

_REASONING_KEYWORDS = frozenset({
    "think through", "reason about", "explain why",
    "what are the tradeoffs", "pros and cons",
    "implications of",
})

_REASONING_PATTERNS = [
    r"how\s+does\b.*\bwork",
    r"how\s+do\b.*\bwork",
    r"explain\b.*\bin\s+detail",
    r"what\s+are\s+the\s+(security|performance|tradeoff|design)\s+implications",
]

_ADVERSARIAL_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+\w+\.\s+do\s+anything",
    r"pretend\s+you\s+have\s+no\s+restrictions",
    r"show\s+me\s+your\s+(system\s+)?prompt",
    r"act\s+as\s+root",
    r"ignore\s+safety",
    r"rm\s+-rf",
]


def _has_pattern(text: str, keywords: frozenset[str]) -> tuple[bool, int]:
    """Check if text contains any keyword pattern. Returns (matched, count)."""
    text_lower = text.lower()
    count = sum(1 for kw in keywords if kw in text_lower)
    return count > 0, count


def _has_regex(text: str, patterns: list[str]) -> bool:
    """Check if text matches any regex pattern."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _count_sentences(text: str) -> int:
    """Rough sentence count."""
    return max(1, len(re.split(r'[.!?]+', text.strip())) - 1) or 1


def _estimate_output_length(text: str, intent: IntentType) -> OutputLength:
    """Estimate expected output length based on prompt characteristics."""
    text_lower = text.lower()

    # Long-form markers
    has_long, _ = _has_pattern(text, _LONG_FORM_KEYWORDS)
    if has_long:
        return OutputLength.LONG

    # Multi-step and complex delegation tend to produce medium-long output
    if intent in (IntentType.COMPLEX_DELEGATION, IntentType.MULTI_STEP):
        return OutputLength.MEDIUM

    # Tool use with synthesis
    if intent == IntentType.TOOL_USE:
        return OutputLength.MEDIUM

    # Short prompts → short answers
    if len(text.split()) < 10:
        return OutputLength.SHORT

    # Medium-length questions about technical topics
    if any(w in text_lower for w in ("explain", "describe", "compare", "how does")):
        return OutputLength.MEDIUM

    return OutputLength.SHORT


def classify(message: str) -> ClassificationResult:
    """Classify a user message into intent, confidence, and complexity.

    Uses heuristic keyword matching for Sprint 1.2. Will be enhanced with
    LLM-based classification in later sprints when Ollama classifier is tuned.

    Args:
        message: The raw user message text.

    Returns:
        ClassificationResult with all routing signals.
    """
    if not message or not message.strip():
        return ClassificationResult(
            intent=IntentType.DIRECT_CHAT,
            confidence=0.5,
            complexity=ComplexityTier.SIMPLE,
            expected_output_length=OutputLength.SHORT,
            uses_graph=False,
        )

    text = message.strip()
    text_lower = text.lower()
    word_count = len(text.split())

    # --- Check for adversarial patterns first (D4: safe path) ---
    if _has_regex(text, _ADVERSARIAL_PATTERNS):
        logger.warning("intent.adversarial_detected", message_preview=text[:50])
        return ClassificationResult(
            intent=IntentType.DIRECT_CHAT,
            confidence=0.95,
            complexity=ComplexityTier.SIMPLE,
            expected_output_length=OutputLength.SHORT,
            uses_graph=False,
        )

    # --- Detect multi-step patterns ---
    has_multi, multi_count = _has_pattern(text, _MULTI_STEP_MARKERS)
    # Also check regex patterns like "first.*then"
    has_first_then = bool(re.search(r"first\b.*\bthen\b", text_lower))

    # Also check for "X, then Y" pattern (comma + then)
    has_comma_then = bool(re.search(r",\s*then\b", text_lower))

    if (has_multi and multi_count >= 2) or has_first_then or has_comma_then:
        output_len = _estimate_output_length(text, IntentType.MULTI_STEP)
        return ClassificationResult(
            intent=IntentType.MULTI_STEP,
            confidence=0.80,
            complexity=ComplexityTier.COMPLEX,
            expected_output_length=output_len,
            uses_graph=True,
        )

    # --- Detect long-form generation requests ---
    has_long, _ = _has_pattern(text, _LONG_FORM_KEYWORDS)
    if has_long:
        return ClassificationResult(
            intent=IntentType.DIRECT_CHAT,
            confidence=0.90,
            complexity=ComplexityTier.HEAVY,
            expected_output_length=OutputLength.LONG,
            uses_graph=True,
        )

    # --- Detect complex delegation ---
    has_complex, complex_count = _has_pattern(text, _COMPLEX_KEYWORDS)
    has_tool, tool_count = _has_pattern(text, _TOOL_KEYWORDS)

    # Multi-action signals: ", and " or multiple sentences suggest complex delegation
    has_multi_action = ", and " in text_lower or _count_sentences(text) >= 2
    if has_complex and (complex_count >= 2 or has_tool or (has_complex and has_multi_action)):
        output_len = _estimate_output_length(text, IntentType.COMPLEX_DELEGATION)
        return ClassificationResult(
            intent=IntentType.COMPLEX_DELEGATION,
            confidence=0.75,
            complexity=ComplexityTier.COMPLEX,
            expected_output_length=output_len,
            uses_graph=True,
        )

    # --- Detect tool use ---
    if has_tool and tool_count >= 1:
        return ClassificationResult(
            intent=IntentType.TOOL_USE,
            confidence=0.85,
            complexity=ComplexityTier.MEDIUM,
            expected_output_length=OutputLength.MEDIUM,
            uses_graph=True,
        )

    # --- Detect knowledge retrieval ---
    has_knowledge, _ = _has_pattern(text, _KNOWLEDGE_KEYWORDS)
    if has_knowledge:
        # Simple knowledge recall vs synthesis
        if word_count > 20 or "how does it relate" in text_lower or "compare" in text_lower:
            return ClassificationResult(
                intent=IntentType.KNOWLEDGE_RETRIEVAL,
                confidence=0.75,
                complexity=ComplexityTier.MEDIUM,
                expected_output_length=OutputLength.MEDIUM,
                uses_graph=True,
            )
        return ClassificationResult(
            intent=IntentType.KNOWLEDGE_RETRIEVAL,
            confidence=0.90,
            complexity=ComplexityTier.SIMPLE,
            expected_output_length=OutputLength.SHORT,
            uses_graph=False,
        )

    # --- Medium complexity direct_chat (reasoning/analysis questions) ---
    has_reasoning, _ = _has_pattern(text, _REASONING_KEYWORDS)
    has_reasoning = has_reasoning or _has_regex(text, _REASONING_PATTERNS)
    if has_reasoning or word_count > 20:
        confidence = 0.70 if has_reasoning else 0.65
        return ClassificationResult(
            intent=IntentType.DIRECT_CHAT,
            confidence=confidence,
            complexity=ComplexityTier.MEDIUM,
            expected_output_length=OutputLength.MEDIUM,
            uses_graph=True,
        )

    # --- Default: simple direct_chat (fast-path) ---
    confidence = 0.95 if word_count < 8 else 0.88
    return ClassificationResult(
        intent=IntentType.DIRECT_CHAT,
        confidence=confidence,
        complexity=ComplexityTier.SIMPLE,
        expected_output_length=OutputLength.SHORT,
        uses_graph=False,
    )
