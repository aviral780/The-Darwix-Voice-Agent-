"""LLM access layer.

Everything that talks to a language model goes through here so that retries,
timing and provider failover are handled in one place instead of being
repeated in the agent, the signal extractor and the evaluation scripts.

Failover exists because the whole build runs on a free tier. Groq's free tier
is rate limited per minute, and a 429 in the middle of a recorded demo would
be unrecoverable. If a Gemini key is present the call transparently fails over
rather than raising.
"""

from __future__ import annotations

import json
import time

import httpx
from groq import Groq
from groq import APIStatusError, RateLimitError

from core import config
from core.timing import LatencyRecorder

_client: Groq | None = None

# Retry on transient failures only. A 400 means a malformed request and retrying
# it just wastes the rate-limit budget.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 0.6


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.require_groq_key())
    return _client


class LLMError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 600,
    json_mode: bool = False,
    recorder: LatencyRecorder | None = None,
    stage: str = "llm",
) -> str:
    """Send a chat completion and return the text.

    Falls back to Gemini if Groq is rate limited and a Gemini key is configured.
    """
    model = model or config.CHAT_MODEL
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            if recorder is not None:
                with recorder.span(stage, model=model, attempt=attempt):
                    response = client().chat.completions.create(**kwargs)
            else:
                response = client().chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()

        except (RateLimitError, APIStatusError) as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            if status is not None and status not in RETRYABLE_STATUS:
                raise LLMError(f"Groq returned {status}: {exc}") from exc
            # Exponential backoff, but only if another attempt is coming.
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))

        except Exception as exc:  # network-level failure
            last_error = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))

    if config.GEMINI_API_KEY:
        try:
            return _gemini_chat(messages, temperature, max_tokens, json_mode)
        except Exception as exc:
            raise LLMError(f"Groq and Gemini both failed. Last Groq error: {last_error}") from exc

    raise LLMError(
        f"Groq failed after {MAX_ATTEMPTS} attempts and no GEMINI_API_KEY fallback "
        f"is configured. Last error: {last_error}"
    )


def chat_json(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
    recorder: LatencyRecorder | None = None,
    stage: str = "llm",
) -> dict:
    """Chat completion that must return a JSON object.

    Signal extraction runs on every audio chunk, so a single malformed response
    must not take the pipeline down. On a parse failure this returns {} and lets
    the caller treat it as "no signal detected" rather than raising.
    """
    raw = chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
        recorder=recorder,
        stage=stage,
    )
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _gemini_chat(messages: list[dict], temperature: float, max_tokens: int, json_mode: bool) -> str:
    """Minimal Gemini REST fallback.

    Gemini has no system role, so the system prompt is folded into the first
    user turn. This path is a safety net for rate limits, not the main route.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    if system_parts and contents:
        contents[0]["parts"][0]["text"] = (
            "\n\n".join(system_parts) + "\n\n" + contents[0]["parts"][0]["text"]
        )

    generation_config = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={config.GEMINI_API_KEY}"
    )
    response = httpx.post(
        url,
        json={"contents": contents, "generationConfig": generation_config},
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
