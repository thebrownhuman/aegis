"""Project Aegis — Context window manager.

Sprint 1.2 Task 11: Manages token counting, budget allocation per role,
conversation history truncation, and auto-summarization triggers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader

logger = structlog.get_logger(__name__)

# Lazy-load tiktoken to avoid import cost when not needed
_encoding = None


def _get_encoding():
    """Get tiktoken encoding, lazily loaded."""
    global _encoding
    if _encoding is None:
        import tiktoken

        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken cl100k_base encoding."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


# --- Token budgets per role (from orchestrator_spec.md §9.1) ---

@dataclass(frozen=True)
class TokenBudget:
    """Token budget allocation for a role."""

    max_input: int
    max_output: int
    knowledge_context_ratio: float = 0.30  # % of remaining for Qdrant context
    tool_results_ratio: float = 0.20  # % of remaining for tool results


BUDGETS: dict[str, TokenBudget] = {
    "owner": TokenBudget(max_input=8192, max_output=4096),
    "named_guest": TokenBudget(max_input=4096, max_output=2048),
    "anonymous_guest": TokenBudget(max_input=2048, max_output=1024),
}


def get_budget(role: str) -> TokenBudget:
    """Get token budget for a role. Falls back to anonymous_guest."""
    return BUDGETS.get(role, BUDGETS["anonymous_guest"])


# --- Prompt template rendering ---

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_system_prompt(role: str, **kwargs: Any) -> str:
    """Render the appropriate system prompt template for a role."""
    template_name = "system_owner.j2" if role == "owner" else "system_guest.j2"
    template = _jinja_env.get_template(template_name)
    return template.render(**kwargs)


def render_template(template_name: str, **kwargs: Any) -> str:
    """Render any prompt template by name."""
    template = _jinja_env.get_template(template_name)
    return template.render(**kwargs)


# --- Message representation ---

@dataclass
class Message:
    """A single message in the conversation context."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0 and self.content:
            self.token_count = count_tokens(self.content)


# --- Context window assembly ---

@dataclass
class ContextWindow:
    """Manages the context window for a single request.

    Assembles system prompt + knowledge context + tool results + conversation
    history within the role's token budget.
    """

    role: str
    budget: TokenBudget = field(init=False)
    system_prompt: str = ""
    knowledge_context: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.budget = get_budget(self.role)

    @property
    def system_tokens(self) -> int:
        return count_tokens(self.system_prompt) if self.system_prompt else 0

    @property
    def remaining_budget(self) -> int:
        """Tokens remaining after system prompt."""
        return max(0, self.budget.max_input - self.system_tokens)

    @property
    def knowledge_budget(self) -> int:
        """Token budget for knowledge context."""
        return int(self.remaining_budget * self.budget.knowledge_context_ratio)

    @property
    def tool_budget(self) -> int:
        """Token budget for tool results."""
        return int(self.remaining_budget * self.budget.tool_results_ratio)

    @property
    def conversation_budget(self) -> int:
        """Token budget for conversation history (what's left after knowledge + tools)."""
        used = 0
        # Actual knowledge tokens used
        for ctx in self.knowledge_context:
            used += count_tokens(ctx)
        # Actual tool tokens used
        for tr in self.tool_results:
            used += count_tokens(tr)
        return max(0, self.remaining_budget - used)

    def set_system_prompt(self, role: str, **kwargs: Any) -> None:
        """Render and set the system prompt."""
        self.system_prompt = render_system_prompt(role, **kwargs)

    def add_knowledge_context(self, context: str) -> bool:
        """Add knowledge context if within budget. Returns True if added."""
        tokens = count_tokens(context)
        current = sum(count_tokens(c) for c in self.knowledge_context)
        if current + tokens <= self.knowledge_budget:
            self.knowledge_context.append(context)
            return True
        logger.debug("context.knowledge_budget_exceeded", tokens=tokens, budget=self.knowledge_budget)
        return False

    def add_tool_result(self, result: str) -> bool:
        """Add tool result if within budget. Returns True if added."""
        tokens = count_tokens(result)
        current = sum(count_tokens(r) for r in self.tool_results)
        if current + tokens <= self.tool_budget:
            self.tool_results.append(result)
            return True
        logger.debug("context.tool_budget_exceeded", tokens=tokens, budget=self.tool_budget)
        return False

    def fit_messages(self, messages: list[Message]) -> list[Message]:
        """Fit conversation messages into remaining budget.

        Keeps most recent messages first, drops oldest if budget exceeded.
        """
        budget = self.conversation_budget
        fitted: list[Message] = []
        total = 0

        # Iterate from most recent to oldest
        for msg in reversed(messages):
            if total + msg.token_count <= budget:
                fitted.append(msg)
                total += msg.token_count
            else:
                break

        fitted.reverse()
        self.messages = fitted

        if len(fitted) < len(messages):
            dropped = len(messages) - len(fitted)
            logger.info("context.messages_truncated", dropped=dropped, kept=len(fitted))

        return fitted

    def assemble(self) -> list[dict[str, str]]:
        """Assemble the final message list for the LLM call.

        Returns list of {role, content} dicts in the format expected by
        LangChain and most LLM APIs.
        """
        result: list[dict[str, str]] = []

        # System prompt
        parts = [self.system_prompt] if self.system_prompt else []

        if self.knowledge_context:
            parts.append("\n## Relevant Knowledge\n" + "\n\n".join(self.knowledge_context))

        if self.tool_results:
            parts.append("\n## Tool Results\n" + "\n\n".join(self.tool_results))

        if parts:
            result.append({"role": "system", "content": "\n".join(parts)})

        # Conversation messages
        for msg in self.messages:
            result.append({"role": msg.role, "content": msg.content})

        return result

    @property
    def total_tokens(self) -> int:
        """Total tokens across all content."""
        total = self.system_tokens
        for ctx in self.knowledge_context:
            total += count_tokens(ctx)
        for tr in self.tool_results:
            total += count_tokens(tr)
        for msg in self.messages:
            total += msg.token_count
        return total


# --- Auto-summarization ---

SUMMARIZATION_THRESHOLD = 10  # Trigger after this many messages


def should_summarize(message_count: int) -> bool:
    """Check if conversation should trigger auto-summarization."""
    return message_count >= SUMMARIZATION_THRESHOLD


def prepare_summarization_prompt(messages: list[Message], max_tokens: int = 200) -> str:
    """Render the summarization prompt for a set of messages."""
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    return render_template("summarization.j2", messages=msg_dicts, max_tokens=max_tokens)
