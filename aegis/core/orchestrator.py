"""Project Aegis — LangGraph orchestrator.

Sprint 1.2 Task 8: Full state machine for complex queries that require
tool use, multi-step planning, or knowledge retrieval with synthesis.

Simple queries are handled by fast_chain.py — this module handles everything
that the fast-path cannot.
"""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Annotated, Any, TypedDict

import structlog
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel

from aegis.core.context import (
    ContextWindow,
    Message,
    count_tokens,
    render_system_prompt,
    should_summarize,
    prepare_summarization_prompt,
)
from aegis.core.intent import (
    ClassificationResult,
    ComplexityTier,
    IntentType,
    classify,
)
from aegis.power.model_router import ModelRouter, RoutingResult

logger = structlog.get_logger(__name__)

# --- Constants ---
MAX_TOOL_ITERATIONS = 5
MAX_PLAN_STEPS = 8
STATIC_FALLBACK_RESPONSE = (
    "I'm temporarily unable to process your request. Please try again in a moment."
)


# --- State definition ---

class OrchestratorState(TypedDict, total=False):
    """State that flows through the LangGraph orchestrator."""

    # Core identifiers
    request_id: str
    user_id: str
    role: str  # "owner" | "named_guest" | "anonymous_guest"
    user_message: str

    # Classification
    classification: dict[str, Any]  # Serialized ClassificationResult
    intent: str
    confidence: float
    complexity: str

    # Routing
    provider_name: str
    model_name: str
    is_fallback: bool

    # Context
    conversation_history: list[dict[str, str]]  # [{role, content}]
    retrieved_context: list[str]  # Qdrant results
    tool_results: list[dict[str, Any]]  # Tool execution results
    plan_steps: list[str]
    current_step: int

    # Output
    response: str
    response_tokens: int

    # Control
    iteration_count: int
    error: str | None
    fallback_attempted: bool


# --- Node functions ---

def classify_node(state: OrchestratorState) -> dict[str, Any]:
    """CLASSIFY: Determine intent, confidence, and complexity."""
    message = state.get("user_message", "")

    try:
        result = classify(message)
        logger.info(
            "orchestrator.classify",
            request_id=state.get("request_id"),
            intent=result.intent.value,
            confidence=result.confidence,
            complexity=result.complexity.value,
            uses_graph=result.uses_graph,
        )
        return {
            "classification": {
                "intent": result.intent.value,
                "confidence": result.confidence,
                "complexity": result.complexity.value,
                "expected_output_length": result.expected_output_length.value,
                "uses_graph": result.uses_graph,
            },
            "intent": result.intent.value,
            "confidence": result.confidence,
            "complexity": result.complexity.value,
        }
    except Exception as e:
        logger.error("orchestrator.classify_error", error=str(e))
        return {
            "error": f"classification_failed: {e}",
            "intent": IntentType.DIRECT_CHAT.value,
            "confidence": 0.5,
            "complexity": ComplexityTier.SIMPLE.value,
        }


def retrieve_node(state: OrchestratorState) -> dict[str, Any]:
    """RETRIEVE: Query Qdrant for relevant knowledge context.

    Placeholder — actual Qdrant integration comes in Sprint 1.3 (knowledge layer).
    For now, returns empty context.
    """
    logger.info(
        "orchestrator.retrieve",
        request_id=state.get("request_id"),
        intent=state.get("intent"),
    )
    # TODO: Sprint 1.3 — integrate Qdrant similarity search
    return {"retrieved_context": []}


def route_model_node(state: OrchestratorState) -> dict[str, Any]:
    """ROUTE_MODEL: Select provider based on complexity tier."""
    complexity_str = state.get("complexity", "simple")
    complexity = ComplexityTier(complexity_str)

    router = _get_router()
    result = router.route(complexity)

    logger.info(
        "orchestrator.route",
        request_id=state.get("request_id"),
        complexity=complexity_str,
        provider=result.provider.name,
        model=result.provider.model,
        is_fallback=result.is_fallback,
    )

    return {
        "provider_name": result.provider.name,
        "model_name": result.provider.model,
        "is_fallback": result.is_fallback,
    }


def plan_node(state: OrchestratorState) -> dict[str, Any]:
    """PLAN: Decompose complex tasks into steps.

    Placeholder — full planning with LLM comes in later sprints.
    For now, treats the entire request as a single generation step.
    """
    logger.info(
        "orchestrator.plan",
        request_id=state.get("request_id"),
        intent=state.get("intent"),
    )
    return {
        "plan_steps": ["generate_response"],
        "current_step": 0,
    }


def execute_tool_node(state: OrchestratorState) -> dict[str, Any]:
    """EXECUTE_TOOL: Run a tool and collect results.

    Placeholder — tool execution framework comes in Sprint 1.4.
    """
    iteration = state.get("iteration_count", 0) + 1

    logger.info(
        "orchestrator.execute_tool",
        request_id=state.get("request_id"),
        iteration=iteration,
    )

    if iteration >= MAX_TOOL_ITERATIONS:
        logger.warning("orchestrator.tool_loop_limit", iteration=iteration)
        return {
            "iteration_count": iteration,
            "error": "tool_iteration_limit_exceeded",
        }

    # TODO: Sprint 1.4 — integrate tool execution
    return {
        "iteration_count": iteration,
        "tool_results": state.get("tool_results", []),
    }


def generate_node(state: OrchestratorState) -> dict[str, Any]:
    """GENERATE: Call the routed model to produce a response.

    For Sprint 1.2, this produces a placeholder indicating the routing
    decision. Actual LLM calls are wired in when provider integrations
    are tested (requires API keys).
    """
    provider = state.get("provider_name", "ollama")
    model = state.get("model_name", "qwen3:1.7b")
    message = state.get("user_message", "")

    logger.info(
        "orchestrator.generate",
        request_id=state.get("request_id"),
        provider=provider,
        model=model,
    )

    # Build context window
    role = state.get("role", "owner")
    ctx = ContextWindow(role=role)
    ctx.set_system_prompt(role)

    # Add retrieved knowledge context
    for kc in state.get("retrieved_context", []):
        ctx.add_knowledge_context(kc)

    # Add tool results
    for tr in state.get("tool_results", []):
        result_str = str(tr.get("result", ""))
        ctx.add_tool_result(result_str)

    # Fit conversation history
    history_dicts = state.get("conversation_history", [])
    history_msgs = [Message(role=h["role"], content=h["content"]) for h in history_dicts]
    ctx.fit_messages(history_msgs)

    # Assemble final messages
    assembled = ctx.assemble()
    assembled.append({"role": "user", "content": message})

    # For Sprint 1.2, we return a structured placeholder showing the routing worked
    # Real LLM invocation wired in when API keys are configured
    response = (
        f"[Aegis Orchestrator] Routed to {provider}/{model} "
        f"(intent={state.get('intent')}, complexity={state.get('complexity')}). "
        f"LLM integration pending API key configuration."
    )

    return {
        "response": response,
        "response_tokens": count_tokens(response),
    }


def learn_node(state: OrchestratorState) -> dict[str, Any]:
    """LEARN: Save useful responses to knowledge base.

    Placeholder — Qdrant write integration comes in Sprint 1.3.
    """
    logger.debug(
        "orchestrator.learn",
        request_id=state.get("request_id"),
        response_tokens=state.get("response_tokens", 0),
    )
    # TODO: Sprint 1.3 — save to Qdrant with enriched metadata (ID-20)
    return {}


def format_node(state: OrchestratorState) -> dict[str, Any]:
    """FORMAT: Apply safety filtering and response formatting."""
    role = state.get("role", "owner")
    response = state.get("response", "")

    # Guest safety filtering (D7: deny-by-default for anonymous)
    if role == "anonymous_guest":
        # Strip any tool results or internal references
        if "[Aegis Orchestrator]" in response:
            response = "I can help answer your question. How can I assist you?"

    logger.info(
        "orchestrator.format",
        request_id=state.get("request_id"),
        role=role,
        response_tokens=count_tokens(response),
    )

    return {"response": response}


def error_node(state: OrchestratorState) -> dict[str, Any]:
    """ERROR: Handle errors with fallback responses."""
    error = state.get("error", "unknown_error")
    fallback_attempted = state.get("fallback_attempted", False)

    logger.error(
        "orchestrator.error",
        request_id=state.get("request_id"),
        error=error,
        fallback_attempted=fallback_attempted,
    )

    if not fallback_attempted:
        # Try Ollama as final fallback
        return {
            "fallback_attempted": True,
            "provider_name": "ollama",
            "model_name": "qwen3:1.7b",
            "error": None,
        }

    # All fallbacks exhausted — return static response (ID-15)
    return {
        "response": STATIC_FALLBACK_RESPONSE,
        "response_tokens": count_tokens(STATIC_FALLBACK_RESPONSE),
    }


# --- Routing functions ---

def route_after_classify(state: OrchestratorState) -> str:
    """Decide next node after classification."""
    if state.get("error"):
        return "error_handler"

    intent = state.get("intent", "direct_chat")
    complexity = state.get("complexity", "simple")

    if intent == "knowledge_retrieval" and complexity != "simple":
        return "retrieve"

    if intent in ("complex_delegation", "multi_step"):
        return "plan"

    if intent == "tool_use":
        return "route_model"

    # Medium+ direct_chat goes through full graph
    if complexity in ("medium", "complex", "heavy"):
        return "route_model"

    # Simple direct_chat shouldn't be here (should use fast-path)
    # But handle gracefully
    return "route_model"


def route_after_retrieve(state: OrchestratorState) -> str:
    """After knowledge retrieval, always route to model selection."""
    return "route_model"


def route_after_plan(state: OrchestratorState) -> str:
    """After planning, decide: tool execution or direct generation."""
    intent = state.get("intent", "")
    if intent in ("tool_use", "multi_step"):
        return "execute_tool"
    return "route_model"


def route_after_tool(state: OrchestratorState) -> str:
    """After tool execution, decide: more tools or generate."""
    if state.get("error"):
        return "error_handler"

    iteration = state.get("iteration_count", 0)
    if iteration >= MAX_TOOL_ITERATIONS:
        return "generate"

    plan_steps = state.get("plan_steps", [])
    current = state.get("current_step", 0)
    if current < len(plan_steps) - 1:
        return "execute_tool"

    return "generate"


def route_after_error(state: OrchestratorState) -> str:
    """After error handling, retry with fallback or respond."""
    # If error was cleared (fallback provider set), retry generation
    if state.get("error") is None and not state.get("response"):
        return "generate"
    # If we have a final response (either from fallback or static), format it
    return "format"


# --- Graph builder ---

def build_orchestrator_graph() -> StateGraph:
    """Build the LangGraph state machine for the orchestrator.

    Returns a compiled graph ready for invocation.
    """
    builder = StateGraph(OrchestratorState)

    # Add nodes
    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("route_model", route_model_node)
    builder.add_node("plan", plan_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("generate", generate_node)
    builder.add_node("learn", learn_node)
    builder.add_node("format", format_node)
    builder.add_node("error_handler", error_node)

    # Set entry point
    builder.set_entry_point("classify")

    # Add conditional edges
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_conditional_edges("retrieve", route_after_retrieve)
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_conditional_edges("execute_tool", route_after_tool)
    builder.add_conditional_edges("error_handler", route_after_error)

    # Fixed edges
    builder.add_edge("route_model", "generate")
    builder.add_edge("generate", "learn")
    builder.add_edge("learn", "format")
    builder.add_edge("format", END)

    return builder.compile()


# --- Module-level singleton ---

_router: ModelRouter | None = None
_graph = None


def _get_router() -> ModelRouter:
    """Get or create the model router singleton."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


def get_graph():
    """Get or create the compiled orchestrator graph."""
    global _graph
    if _graph is None:
        _graph = build_orchestrator_graph()
    return _graph


async def run_orchestrator(
    message: str,
    request_id: str,
    user_id: str,
    role: str = "owner",
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run a message through the full orchestrator graph.

    This is the main entry point for complex queries that need the full
    LangGraph state machine (tool use, multi-step, knowledge synthesis).

    Args:
        message: User's message text.
        request_id: Correlation ID for logging.
        user_id: User identifier.
        role: User role.
        conversation_history: Previous messages [{role, content}].

    Returns:
        Dict with response, provider, model, and metadata.
    """
    start = datetime.now(UTC)
    graph = get_graph()

    initial_state: OrchestratorState = {
        "request_id": request_id,
        "user_id": user_id,
        "role": role,
        "user_message": message,
        "conversation_history": conversation_history or [],
        "retrieved_context": [],
        "tool_results": [],
        "plan_steps": [],
        "current_step": 0,
        "iteration_count": 0,
        "error": None,
        "fallback_attempted": False,
        "response": "",
        "response_tokens": 0,
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
        logger.error(
            "orchestrator.fatal_error",
            request_id=request_id,
            error=str(e),
            latency_ms=elapsed,
        )
        return {
            "response": STATIC_FALLBACK_RESPONSE,
            "provider": "none",
            "model": "none",
            "intent": "error",
            "complexity": "error",
            "latency_ms": elapsed,
            "error": str(e),
        }

    elapsed = (datetime.now(UTC) - start).total_seconds() * 1000

    logger.info(
        "orchestrator.complete",
        request_id=request_id,
        intent=result.get("intent"),
        complexity=result.get("complexity"),
        provider=result.get("provider_name"),
        latency_ms=round(elapsed, 1),
    )

    return {
        "response": result.get("response", STATIC_FALLBACK_RESPONSE),
        "provider": result.get("provider_name", "unknown"),
        "model": result.get("model_name", "unknown"),
        "intent": result.get("intent", "unknown"),
        "complexity": result.get("complexity", "unknown"),
        "latency_ms": elapsed,
        "is_fallback": result.get("is_fallback", False),
    }
