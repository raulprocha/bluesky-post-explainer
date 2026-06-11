"""LLM Router with multi-provider fallback routing via LiteLLM."""

import asyncio
import logging
from dataclasses import dataclass

import litellm

from src.exceptions import AllProvidersFailedError

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o",
    "gemini/gemini-1.5-pro",
]

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0


@dataclass
class LLMResponse:
    """Normalized response from an LLM provider."""

    content: str
    model: str
    usage: dict


class LLMRouter:
    """Routes LLM completion requests with fallback and retry logic.

    When model is explicitly specified, routes to that model only (no fallback).
    When model is None, tries each provider in order, falling back on failure.
    Rate-limited (429) requests are retried with exponential backoff.
    """

    def __init__(
        self, providers: list[str] | None = None, fallback: bool = True
    ) -> None:
        """Initialize with ordered provider list.

        Args:
            providers: Ordered list of model identifiers. Defaults to
                ['claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini/gemini-1.5-pro'].
            fallback: Whether to attempt next provider on failure (when model is None).
        """
        self.providers = providers if providers is not None else list(DEFAULT_PROVIDERS)
        self.fallback = fallback

    async def completion(
        self, messages: list[dict], model: str | None = None, **kwargs
    ) -> LLMResponse:
        """Route completion request to LLM provider(s).

        Args:
            messages: Chat message list in OpenAI format.
            model: Explicit model to use. If None, uses fallback chain.
            **kwargs: Additional arguments passed to litellm.acompletion.

        Returns:
            LLMResponse with content, model name, and usage stats.

        Raises:
            AllProvidersFailedError: When all providers in fallback chain fail.
            Exception: When explicit model fails (re-raises the provider error).
        """
        if model is not None:
            # Explicit provider routing - no fallback (Property 11)
            return await self._call_provider(model, messages, **kwargs)

        # Fallback mode - try each provider in order
        attempts: list[tuple[str, Exception]] = []

        for provider in self.providers:
            try:
                return await self._call_with_retry(provider, messages, **kwargs)
            except Exception as e:
                logger.warning(
                    "Provider %s failed: %s", provider, str(e)
                )
                attempts.append((provider, e))

                if not self.fallback:
                    break

        # All providers failed (Property 14: error includes all provider names)
        raise AllProvidersFailedError(attempts)

    async def _call_with_retry(
        self, model: str, messages: list[dict], **kwargs
    ) -> LLMResponse:
        """Call a provider with exponential backoff retry on rate limits.

        Property 15: For N consecutive transient failures (N<=3), makes exactly
        N+1 total attempts (1 initial + N retries).

        Args:
            model: The model identifier.
            messages: Chat messages.
            **kwargs: Additional litellm arguments.

        Returns:
            LLMResponse on success.

        Raises:
            The last exception encountered if all retries are exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):  # 1 initial + MAX_RETRIES retries = 4 max
            try:
                return await self._call_provider(model, messages, **kwargs)
            except Exception as e:
                last_exception = e
                if not self._is_rate_limit(e):
                    # Non-rate-limit errors are not retried
                    raise
                if attempt < MAX_RETRIES:
                    backoff = BASE_BACKOFF_SECONDS * (2**attempt)
                    logger.info(
                        "Rate limited by %s, retrying in %.1fs (attempt %d/%d)",
                        model,
                        backoff,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(backoff)

        # All retries exhausted
        raise last_exception  # type: ignore[misc]

    async def _call_provider(
        self, model: str, messages: list[dict], **kwargs
    ) -> LLMResponse:
        """Make a single completion call to a provider via litellm.

        Args:
            model: The model identifier.
            messages: Chat messages.
            **kwargs: Additional arguments for litellm.acompletion.

        Returns:
            Normalized LLMResponse.
        """
        response = await litellm.acompletion(
            model=model, messages=messages, **kwargs
        )
        return self._normalize_response(response)

    def _normalize_response(self, response) -> LLMResponse:
        """Normalize a litellm response into our LLMResponse dataclass.

        Args:
            response: Raw response from litellm.acompletion.

        Returns:
            LLMResponse with extracted content, model, and usage.
        """
        content = response.choices[0].message.content or ""
        model = response.model or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(content=content, model=model, usage=usage)

    @staticmethod
    def _is_rate_limit(exception: Exception) -> bool:
        """Check if an exception represents a rate limit (429) error.

        Args:
            exception: The exception to check.

        Returns:
            True if it's a rate-limit error.
        """
        # LiteLLM wraps 429 errors in specific exception types
        if hasattr(exception, "status_code"):
            return getattr(exception, "status_code") == 429
        # Also check string representation as a fallback
        return "429" in str(exception) or "rate" in str(exception).lower()
