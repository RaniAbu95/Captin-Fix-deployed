import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_EFFORT

# Fail fast on 429 quota errors instead of backing off for minutes
# while the HTTP request (and Render's proxy) waits.
_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=2)


def complete(user: str, system: str | None = None, max_tokens: int = 64000):
    """Send one request to Claude and return (text, stop_reason).

    The system prompt is marked for prompt caching so repeated calls that share
    it (one per test case in executor.py) only pay full price for it once.
    Streaming avoids HTTP timeouts on long outputs."""
    kwargs = {}
    if system:
        kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    with _client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        output_config={"effort": CLAUDE_EFFORT},
        # If a safety classifier declines the request, retry it server-side on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
        messages=[{"role": "user", "content": user}],
        **kwargs,
    ) as stream:
        message = stream.get_final_message()
    text = "".join(b.text for b in message.content if b.type == "text")
    return text, message.stop_reason
