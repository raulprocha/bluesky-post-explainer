"""Custom exception hierarchy for the Contextual Post Explainer agent."""


class ExplainerError(Exception):
    """Base exception for the agent."""


class PostNotFoundError(ExplainerError):
    """Post does not exist or was deleted."""


class PostBlockedError(ExplainerError):
    """Post is blocked due to moderation."""


class NetworkError(ExplainerError):
    """Network connectivity issue."""


class SearchError(ExplainerError):
    """Search API failure."""


class AllProvidersFailedError(ExplainerError):
    """All LLM providers failed."""

    def __init__(self, attempts: list[tuple[str, Exception]]):
        self.attempts = attempts
        providers = ", ".join(p for p, _ in attempts)
        super().__init__(f"All providers failed: {providers}")


class GenerationError(ExplainerError):
    """LLM generation produced invalid output."""
