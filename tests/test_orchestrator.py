"""Tests for the LangGraph orchestrator.

Sprint 1.2 Task 8: Validates graph construction, state transitions,
node routing, and end-to-end orchestration.
"""

from __future__ import annotations

import pytest

from aegis.core.orchestrator import (
    MAX_TOOL_ITERATIONS,
    STATIC_FALLBACK_RESPONSE,
    OrchestratorState,
    build_orchestrator_graph,
    classify_node,
    error_node,
    format_node,
    generate_node,
    plan_node,
    retrieve_node,
    route_after_classify,
    route_after_error,
    route_after_plan,
    route_after_tool,
    route_model_node,
    run_orchestrator,
)


class TestGraphConstruction:
    """Test that the orchestrator graph builds correctly."""

    def test_graph_compiles(self) -> None:
        graph = build_orchestrator_graph()
        assert graph is not None

    def test_graph_has_nodes(self) -> None:
        """Graph should contain all expected nodes."""
        graph = build_orchestrator_graph()
        # LangGraph compiled graph exposes node names
        node_names = set(graph.nodes.keys())
        expected = {"classify", "retrieve", "route_model", "plan",
                    "execute_tool", "generate", "learn", "format", "error_handler"}
        # __start__ and __end__ are internal LangGraph nodes
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


class TestClassifyNode:
    """Test the classification node."""

    def test_simple_message(self) -> None:
        state: OrchestratorState = {
            "user_message": "Hello!",
            "request_id": "test-1",
        }
        result = classify_node(state)
        assert result["intent"] == "direct_chat"
        assert result["complexity"] == "simple"
        assert result["confidence"] > 0

    def test_tool_message(self) -> None:
        state: OrchestratorState = {
            "user_message": "Search the web for Python tutorials",
            "request_id": "test-2",
        }
        result = classify_node(state)
        assert result["intent"] == "tool_use"
        assert result["complexity"] == "medium"

    def test_empty_message(self) -> None:
        state: OrchestratorState = {
            "user_message": "",
            "request_id": "test-3",
        }
        result = classify_node(state)
        assert result["intent"] == "direct_chat"


class TestRouting:
    """Test conditional routing functions."""

    def test_simple_chat_routes_to_model(self) -> None:
        state: OrchestratorState = {
            "intent": "direct_chat",
            "complexity": "medium",
            "error": None,
        }
        assert route_after_classify(state) == "route_model"

    def test_knowledge_routes_to_retrieve(self) -> None:
        state: OrchestratorState = {
            "intent": "knowledge_retrieval",
            "complexity": "medium",
            "error": None,
        }
        assert route_after_classify(state) == "retrieve"

    def test_complex_routes_to_plan(self) -> None:
        state: OrchestratorState = {
            "intent": "complex_delegation",
            "complexity": "complex",
            "error": None,
        }
        assert route_after_classify(state) == "plan"

    def test_multi_step_routes_to_plan(self) -> None:
        state: OrchestratorState = {
            "intent": "multi_step",
            "complexity": "complex",
            "error": None,
        }
        assert route_after_classify(state) == "plan"

    def test_tool_use_routes_to_model(self) -> None:
        state: OrchestratorState = {
            "intent": "tool_use",
            "complexity": "medium",
            "error": None,
        }
        assert route_after_classify(state) == "route_model"

    def test_error_routes_to_handler(self) -> None:
        state: OrchestratorState = {
            "intent": "direct_chat",
            "complexity": "simple",
            "error": "classification_failed",
        }
        assert route_after_classify(state) == "error_handler"

    def test_plan_with_tools(self) -> None:
        state: OrchestratorState = {"intent": "tool_use"}
        assert route_after_plan(state) == "execute_tool"

    def test_plan_without_tools(self) -> None:
        state: OrchestratorState = {"intent": "direct_chat"}
        assert route_after_plan(state) == "route_model"

    def test_tool_iteration_limit(self) -> None:
        state: OrchestratorState = {
            "iteration_count": MAX_TOOL_ITERATIONS,
            "plan_steps": [],
            "current_step": 0,
            "error": None,
        }
        assert route_after_tool(state) == "generate"


class TestNodes:
    """Test individual node functions."""

    def test_retrieve_returns_empty_context(self) -> None:
        """Sprint 1.2 placeholder — Qdrant not yet integrated."""
        state: OrchestratorState = {"request_id": "test"}
        result = retrieve_node(state)
        assert result["retrieved_context"] == []

    def test_route_model_selects_provider(self) -> None:
        state: OrchestratorState = {
            "complexity": "simple",
            "request_id": "test",
        }
        result = route_model_node(state)
        assert "provider_name" in result
        assert "model_name" in result

    def test_plan_creates_steps(self) -> None:
        state: OrchestratorState = {
            "intent": "complex_delegation",
            "request_id": "test",
        }
        result = plan_node(state)
        assert len(result["plan_steps"]) > 0

    def test_generate_produces_response(self) -> None:
        state: OrchestratorState = {
            "user_message": "Hello",
            "role": "owner",
            "intent": "direct_chat",
            "complexity": "simple",
            "provider_name": "ollama",
            "model_name": "qwen3:1.7b",
            "request_id": "test",
            "retrieved_context": [],
            "tool_results": [],
            "conversation_history": [],
        }
        result = generate_node(state)
        assert result["response"]
        assert result["response_tokens"] > 0

    def test_format_owner_passthrough(self) -> None:
        state: OrchestratorState = {
            "role": "owner",
            "response": "Test response",
            "request_id": "test",
        }
        result = format_node(state)
        assert result["response"] == "Test response"

    def test_format_anonymous_guest_filtered(self) -> None:
        state: OrchestratorState = {
            "role": "anonymous_guest",
            "response": "[Aegis Orchestrator] internal routing info",
            "request_id": "test",
        }
        result = format_node(state)
        assert "[Aegis Orchestrator]" not in result["response"]

    def test_error_first_attempt_fallback(self) -> None:
        state: OrchestratorState = {
            "error": "provider_timeout",
            "fallback_attempted": False,
            "request_id": "test",
        }
        result = error_node(state)
        assert result["fallback_attempted"] is True
        assert result["provider_name"] == "ollama"

    def test_error_all_exhausted(self) -> None:
        state: OrchestratorState = {
            "error": "all_failed",
            "fallback_attempted": True,
            "request_id": "test",
        }
        result = error_node(state)
        assert result["response"] == STATIC_FALLBACK_RESPONSE


class TestErrorFallbackPath:
    """Test the error → fallback → generate retry path (Codex fix #1)."""

    def test_first_error_routes_to_generate(self) -> None:
        """When error_node clears the error and sets fallback provider,
        route_after_error should route to 'generate' for retry."""
        # Simulate state AFTER error_node runs on first error
        state: OrchestratorState = {
            "error": None,  # error_node cleared this
            "fallback_attempted": True,  # error_node set this
            "provider_name": "ollama",  # error_node set fallback
            "model_name": "qwen3:1.7b",
            "response": "",  # no response yet
        }
        assert route_after_error(state) == "generate"

    def test_final_static_response_routes_to_format(self) -> None:
        """When all fallbacks are exhausted and static response is set,
        route_after_error should route to 'format'."""
        state: OrchestratorState = {
            "error": "all_failed",
            "fallback_attempted": True,
            "response": STATIC_FALLBACK_RESPONSE,
        }
        assert route_after_error(state) == "format"

    def test_error_with_no_response_routes_to_format(self) -> None:
        """When error persists but no retry possible, route to format
        which will use the static fallback."""
        state: OrchestratorState = {
            "error": "persistent_error",
            "fallback_attempted": True,
            "response": STATIC_FALLBACK_RESPONSE,
        }
        assert route_after_error(state) == "format"

    def test_successful_response_after_fallback_routes_to_format(self) -> None:
        """After fallback generate succeeds, route to format."""
        state: OrchestratorState = {
            "error": None,
            "fallback_attempted": True,
            "response": "Some actual response from fallback model",
        }
        assert route_after_error(state) == "format"


class TestEndToEnd:
    """Test full orchestrator pipeline."""

    @pytest.mark.asyncio
    async def test_simple_message(self) -> None:
        result = await run_orchestrator(
            message="Hello!",
            request_id="e2e-1",
            user_id="owner",
            role="owner",
        )
        assert "response" in result
        assert result["intent"] == "direct_chat"
        assert result["response"]

    @pytest.mark.asyncio
    async def test_tool_use_message(self) -> None:
        result = await run_orchestrator(
            message="Search the web for Python tutorials",
            request_id="e2e-2",
            user_id="owner",
            role="owner",
        )
        assert result["intent"] == "tool_use"
        assert result["complexity"] == "medium"

    @pytest.mark.asyncio
    async def test_complex_message(self) -> None:
        result = await run_orchestrator(
            message="Research microservices patterns, identify which ones apply, and create an implementation plan",
            request_id="e2e-3",
            user_id="owner",
            role="owner",
        )
        assert result["intent"] in ("complex_delegation", "multi_step")
        assert result["complexity"] == "complex"

    @pytest.mark.asyncio
    async def test_with_conversation_history(self) -> None:
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        result = await run_orchestrator(
            message="Tell me more",
            request_id="e2e-4",
            user_id="owner",
            role="owner",
            conversation_history=history,
        )
        assert "response" in result

    @pytest.mark.asyncio
    async def test_guest_message(self) -> None:
        result = await run_orchestrator(
            message="Hello!",
            request_id="e2e-5",
            user_id="guest-1",
            role="anonymous_guest",
        )
        assert "response" in result
        # Anonymous guest should get filtered response
        assert "[Aegis Orchestrator]" not in result["response"]
