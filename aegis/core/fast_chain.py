"""Project Aegis — Fast-path LangChain chain.

Sprint 1.2 Task 12a: Lightweight chain for simple queries that bypasses
the full LangGraph state machine. Targets ~60-70% of traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from aegis.core.context import ContextWindow, Message, count_tokens
from aegis.core.intent import ClassificationResult, ComplexityTier

logger = structlog.get_logger(__name__)

# Fast-path constraints (from orchestrator_spec.md §5.3)
MAX_INPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 1000


@dataclass
class FastPathResult:
    """Result from fast-path chain execution."""

    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    escalated: bool = False  # True if we need to escalate to full graph
    escalation_reason: str | None = None


class FastPathChain:
    """Lightweight LangChain chain for simple queries.

    Handles direct_chat and knowledge_retrieval intents with high confidence.
    If anything goes wrong or the query turns out to be complex, escalates
    to the full LangGraph orchestrator.
    """

    def __init__(self, model: BaseChatModel, provider_name: str, model_name: str) -> None:
        self._model = model
        self._provider_name = provider_name
        self._model_name = model_name

    async def run(
        self,
        message: str,
        classification: ClassificationResult,
        role: str = "owner",
        conversation_history: list[Message] | None = None,
        knowledge_context: list[str] | None = None,
        **prompt_kwargs: Any,
    ) -> FastPathResult:
        """Execute the fast-path chain.

        Args:
            message: User's message text.
            classification: Pre-computed classification result.
            role: User role for prompt selection.
            conversation_history: Previous messages for context.
            knowledge_context: Qdrant retrieval results (if knowledge_retrieval intent).
            **prompt_kwargs: Additional template variables.

        Returns:
            FastPathResult with response or escalation signal.
        """
        start = datetime.now(UTC)

        # Build context window
        ctx = ContextWindow(role=role)
        ctx.set_system_prompt(role, **prompt_kwargs)

        # Add knowledge context if available
        if knowledge_context:
            for kc in knowledge_context:
                ctx.add_knowledge_context(kc)

        # Fit conversation history
        history = conversation_history or []
        ctx.fit_messages(history)

        # Check total input tokens
        if ctx.total_tokens > MAX_INPUT_TOKENS:
            logger.info("fast_path.input_too_large", tokens=ctx.total_tokens, max=MAX_INPUT_TOKENS)
            return FastPathResult(
                content="",
                provider=self._provider_name,
                model=self._model_name,
                input_tokens=ctx.total_tokens,
                output_tokens=0,
                latency_ms=0,
                escalated=True,
                escalation_reason="input_tokens_exceeded",
            )

        # Assemble messages for LLM
        assembled = ctx.assemble()
        # Add the current user message
        assembled.append({"role": "user", "content": message})

        # Convert to LangChain message format
        lc_messages = []
        for msg in assembled:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        # Call the model
        try:
            response = await self._model.ainvoke(lc_messages)
        except Exception as e:
            elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
            logger.error(
                "fast_path.model_error",
                provider=self._provider_name,
                error=str(e),
                latency_ms=elapsed,
            )
            return FastPathResult(
                content="",
                provider=self._provider_name,
                model=self._model_name,
                input_tokens=ctx.total_tokens,
                output_tokens=0,
                latency_ms=elapsed,
                escalated=True,
                escalation_reason=f"model_error: {e}",
            )

        content = response.content if isinstance(response.content, str) else str(response.content)
        output_tokens = count_tokens(content)
        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000

        # Check if model is requesting tool calls (escalation trigger)
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info("fast_path.tool_call_requested", escalating=True)
            return FastPathResult(
                content=content,
                provider=self._provider_name,
                model=self._model_name,
                input_tokens=ctx.total_tokens,
                output_tokens=output_tokens,
                latency_ms=elapsed,
                escalated=True,
                escalation_reason="tool_call_requested",
            )

        # Check output token limit
        if output_tokens > MAX_OUTPUT_TOKENS:
            logger.info(
                "fast_path.output_exceeded",
                tokens=output_tokens,
                max=MAX_OUTPUT_TOKENS,
            )
            # Still return the result — it's usable, just long

        logger.info(
            "fast_path.complete",
            provider=self._provider_name,
            input_tokens=ctx.total_tokens,
            output_tokens=output_tokens,
            latency_ms=round(elapsed, 1),
        )

        return FastPathResult(
            content=content,
            provider=self._provider_name,
            model=self._model_name,
            input_tokens=ctx.total_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
        )
