"""OpenAI-compatible provider abstraction."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from toks_bench.security import validate_base_url


@dataclass(frozen=True)
class Provider:
    """A backend that exposes an OpenAI-compatible chat completions endpoint."""

    name: str
    kind: str
    base_url: str
    model: str


@dataclass(frozen=True)
class ToolConfig:
    """Tool-calling configuration passed to the provider."""

    tools: list[dict[str, object]]
    tool_choice: object = "auto"


def create_client(provider: Provider) -> OpenAI:
    """Create an OpenAI client pointing at the provider's base URL."""
    # Re-validate at client creation time so direct programmatic use of Provider
    # objects is also protected.
    base_url = validate_base_url(provider.base_url)
    return OpenAI(base_url=base_url, api_key="dummy", timeout=30.0)


def complete_stream(
    client: OpenAI,
    provider: Provider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[tuple[str, bool, int | None, int | None, str | None], None, None]:
    """Stream a chat completion and yield per-chunk metadata.

    Yields ``(delta_text, is_first_chunk, usage_tokens, prompt_tokens, finish_reason)``.
    ``usage_tokens`` and ``prompt_tokens`` are only non-None when the chunk includes
    usage information. Some servers (e.g. llama-server) omit usage entirely; callers
    should count deltas.
    """
    response = client.chat.completions.create(
        model=provider.model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    is_first = True
    usage: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    for chunk in response:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        raw = choice.delta
        delta = (
            (raw.content or "")
            + (getattr(raw, "reasoning_content", None) or "")
            + (getattr(raw, "reasoning", None) or "")
        )
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage.completion_tokens  # type: ignore[attr-defined]
            prompt_tokens = chunk.usage.prompt_tokens  # type: ignore[attr-defined]
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
        yield delta, is_first, usage, prompt_tokens, finish_reason
        is_first = False
        usage = None
        prompt_tokens = None


def complete_tool(
    client: OpenAI,
    provider: Provider,
    messages: list[dict[str, str]],
    *,
    tools: ToolConfig,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[dict[str, object]], int | None, int | None, str | None]:
    """Send a non-streaming tool-call request and return the parsed tool calls."""
    response = client.chat.completions.create(
        model=provider.model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        tools=cast(list[ChatCompletionToolParam], tools.tools),
        tool_choice=cast(ChatCompletionToolChoiceOptionParam, tools.tool_choice),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    choice = response.choices[0]
    calls: list[dict[str, object]] = []
    if choice.message.tool_calls:
        for call in choice.message.tool_calls:
            # OpenAI may return custom tool calls without a function field.
            if not hasattr(call, "function"):
                continue
            function_call = cast(ChatCompletionMessageToolCall, call).function
            calls.append(
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": function_call.name,
                        "arguments": function_call.arguments,
                    },
                }
            )
    usage = getattr(response, "usage", None)
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    if usage is not None:
        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
    return calls, completion_tokens, prompt_tokens, choice.finish_reason
