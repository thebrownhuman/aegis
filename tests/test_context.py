"""Tests for the context window manager.

Sprint 1.2 Task 11: Validates token counting, budget allocation, message
truncation, and prompt template rendering.
"""

from __future__ import annotations

import pytest

from aegis.core.context import (
    BUDGETS,
    ContextWindow,
    Message,
    count_tokens,
    get_budget,
    render_system_prompt,
    render_template,
    should_summarize,
    prepare_summarization_prompt,
)


class TestTokenCounting:
    """Test token counting utility."""

    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_simple_text(self) -> None:
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Should be ~4 tokens

    def test_longer_text(self) -> None:
        text = "This is a longer piece of text that should have more tokens."
        tokens = count_tokens(text)
        assert tokens > 5

    def test_consistency(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) == count_tokens(text)  # Deterministic


class TestTokenBudgets:
    """Test role-based token budget allocation."""

    def test_owner_budget(self) -> None:
        budget = get_budget("owner")
        assert budget.max_input == 8192
        assert budget.max_output == 4096

    def test_named_guest_budget(self) -> None:
        budget = get_budget("named_guest")
        assert budget.max_input == 4096
        assert budget.max_output == 2048

    def test_anonymous_guest_budget(self) -> None:
        budget = get_budget("anonymous_guest")
        assert budget.max_input == 2048
        assert budget.max_output == 1024

    def test_unknown_role_fallback(self) -> None:
        budget = get_budget("unknown_role")
        assert budget.max_input == 2048  # Falls back to anonymous_guest

    def test_budget_ratios(self) -> None:
        budget = get_budget("owner")
        assert budget.knowledge_context_ratio == 0.30
        assert budget.tool_results_ratio == 0.20


class TestMessage:
    """Test Message dataclass."""

    def test_auto_token_count(self) -> None:
        msg = Message(role="user", content="Hello world")
        assert msg.token_count > 0

    def test_explicit_token_count(self) -> None:
        msg = Message(role="user", content="Hello", token_count=42)
        assert msg.token_count == 42

    def test_empty_content(self) -> None:
        msg = Message(role="user", content="")
        assert msg.token_count == 0


class TestContextWindow:
    """Test context window assembly and budget management."""

    def test_basic_creation(self) -> None:
        ctx = ContextWindow(role="owner")
        assert ctx.budget.max_input == 8192

    def test_system_prompt_rendering(self) -> None:
        ctx = ContextWindow(role="owner")
        ctx.set_system_prompt("owner", user_name="Shivansh")
        assert "Aegis" in ctx.system_prompt
        assert "Shivansh" in ctx.system_prompt
        assert ctx.system_tokens > 0

    def test_guest_system_prompt(self) -> None:
        ctx = ContextWindow(role="anonymous_guest")
        ctx.set_system_prompt("anonymous_guest", owner_name="Shivansh")
        assert "professional" in ctx.system_prompt.lower()
        assert "anonymous" in ctx.system_prompt.lower()

    def test_knowledge_context_within_budget(self) -> None:
        ctx = ContextWindow(role="owner")
        ctx.set_system_prompt("owner")
        added = ctx.add_knowledge_context("Some relevant knowledge about Python.")
        assert added is True
        assert len(ctx.knowledge_context) == 1

    def test_tool_result_within_budget(self) -> None:
        ctx = ContextWindow(role="owner")
        ctx.set_system_prompt("owner")
        added = ctx.add_tool_result("Tool returned: success")
        assert added is True
        assert len(ctx.tool_results) == 1

    def test_message_truncation(self) -> None:
        ctx = ContextWindow(role="anonymous_guest")  # Smallest budget
        ctx.set_system_prompt("anonymous_guest")

        # Create many messages that exceed budget
        messages = [
            Message(role="user", content=f"Message number {i} " * 20)
            for i in range(50)
        ]

        fitted = ctx.fit_messages(messages)
        assert len(fitted) < len(messages)
        assert len(fitted) > 0
        # Most recent messages should be kept
        assert fitted[-1].content == messages[-1].content

    def test_assemble_basic(self) -> None:
        ctx = ContextWindow(role="owner")
        ctx.set_system_prompt("owner")
        ctx.fit_messages([
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ])
        assembled = ctx.assemble()
        assert len(assembled) >= 3  # system + 2 messages
        assert assembled[0]["role"] == "system"
        assert assembled[1]["role"] == "user"
        assert assembled[2]["role"] == "assistant"

    def test_assemble_with_knowledge_and_tools(self) -> None:
        ctx = ContextWindow(role="owner")
        ctx.set_system_prompt("owner")
        ctx.add_knowledge_context("Knowledge: Python is great")
        ctx.add_tool_result("Search result: LangChain docs")
        ctx.fit_messages([Message(role="user", content="Tell me more")])
        assembled = ctx.assemble()
        system_content = assembled[0]["content"]
        assert "Knowledge" in system_content
        assert "Tool Results" in system_content

    def test_total_tokens(self) -> None:
        ctx = ContextWindow(role="owner")
        ctx.set_system_prompt("owner")
        ctx.fit_messages([Message(role="user", content="Hello")])
        assert ctx.total_tokens > 0


class TestPromptTemplates:
    """Test Jinja2 template rendering."""

    def test_render_owner_prompt(self) -> None:
        prompt = render_system_prompt("owner", user_name="Shivansh")
        assert "Aegis" in prompt
        assert "Shivansh" in prompt

    def test_render_guest_prompt(self) -> None:
        prompt = render_system_prompt("named_guest", owner_name="Shivansh", guest_name="Alice")
        assert "Alice" in prompt
        assert "professional" in prompt.lower()

    def test_render_routing_template(self) -> None:
        rendered = render_template("routing.j2", message="Hello world")
        assert "Hello world" in rendered
        assert "intent" in rendered.lower()

    def test_render_summarization_template(self) -> None:
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        rendered = render_template("summarization.j2", messages=messages, max_tokens=200)
        assert "Python" in rendered


class TestAutoSummarization:
    """Test auto-summarization trigger logic."""

    def test_below_threshold(self) -> None:
        assert not should_summarize(5)
        assert not should_summarize(9)

    def test_at_threshold(self) -> None:
        assert should_summarize(10)

    def test_above_threshold(self) -> None:
        assert should_summarize(15)
        assert should_summarize(100)

    def test_prepare_prompt(self) -> None:
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]
        prompt = prepare_summarization_prompt(messages)
        assert "Hello" in prompt
        assert "Hi!" in prompt
