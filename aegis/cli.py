"""Project Aegis — CLI direct chat mode.

Sprint 1.2 Task 12: Interactive CLI for testing the orchestrator pipeline.
Routes messages through intent classification → fast-path or full graph.
"""

from __future__ import annotations

import asyncio
import uuid

import structlog

from aegis.core.intent import classify
from aegis.core.orchestrator import run_orchestrator
from aegis.utils.logger import setup_logging

logger = structlog.get_logger(__name__)


async def chat_loop() -> None:
    """Interactive chat loop for testing the orchestrator."""
    setup_logging()

    print("\n=== Project Aegis — Direct Chat Mode ===")
    print("Type your message and press Enter. Type 'quit' or 'exit' to stop.")
    print("Type '/status' to see routing info for last message.")
    print("Type '/classify <msg>' to see classification without running.\n")

    conversation_history: list[dict[str, str]] = []
    last_result: dict | None = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # --- Special commands ---
        if user_input == "/status":
            if last_result:
                print(f"\n  Intent:     {last_result.get('intent')}")
                print(f"  Complexity: {last_result.get('complexity')}")
                print(f"  Provider:   {last_result.get('provider')}/{last_result.get('model')}")
                print(f"  Fallback:   {last_result.get('is_fallback', False)}")
                print(f"  Latency:    {last_result.get('latency_ms', 0):.0f}ms\n")
            else:
                print("  No message processed yet.\n")
            continue

        if user_input.startswith("/classify "):
            msg = user_input[len("/classify "):]
            result = classify(msg)
            print(f"\n  Intent:      {result.intent.value}")
            print(f"  Confidence:  {result.confidence:.2f}")
            print(f"  Complexity:  {result.complexity.value}")
            print(f"  Output len:  {result.expected_output_length.value}")
            print(f"  Uses graph:  {result.uses_graph}")
            print(f"  Fast-path:   {result.is_simple}\n")
            continue

        # --- Process message through orchestrator ---
        request_id = str(uuid.uuid4())[:8]

        # Classify first to show routing decision
        classification = classify(user_input)
        route_info = (
            f"[{classification.complexity.value}→"
            f"{'graph' if classification.uses_graph else 'fast-path'}]"
        )

        # Run through full orchestrator
        result = await run_orchestrator(
            message=user_input,
            request_id=request_id,
            user_id="owner",
            role="owner",
            conversation_history=conversation_history,
        )
        last_result = result

        # Display response
        print(f"\nAegis {route_info}: {result['response']}\n")

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": result["response"]})

        # Keep history manageable (last 20 messages)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


def main() -> None:
    """Entry point for CLI chat mode."""
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
