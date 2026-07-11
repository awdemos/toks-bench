"""Tests for provider streaming."""


from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from toks_bench.providers import Provider, complete_stream


def _make_chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="id",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=content),
                finish_reason=None,
            )
        ],
        created=0,
        model="m",
        object="chat.completion.chunk",
    )


def test_complete_stream_counts_deltas(monkeypatch) -> None:
    provider = Provider(
        name="test",
        kind="llama-server",
        base_url="http://localhost:8080/v1",
        model="m",
    )

    def fake_create(*args, **kwargs):
        yield _make_chunk("Hello")
        yield _make_chunk(" world")

    client = OpenAI(base_url=provider.base_url, api_key="dummy")
    monkeypatch.setattr(client.chat.completions, "create", fake_create)

    chunks = list(
        complete_stream(
            client,
            provider,
            [{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
        )
    )
    texts = [c[0] for c in chunks]
    first_flags = [c[1] for c in chunks]
    assert texts == ["Hello", " world"]
    assert first_flags == [True, False]
