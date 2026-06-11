"""Port interfaces (abstract base classes) for the hexagonal architecture.

These define the contracts that adapters must implement, allowing
the domain use cases to remain decoupled from infrastructure details.
"""

from abc import ABC, abstractmethod

from src.domain.entities import (
    ExplanationResult,
    ImageEmbed,
    PostContent,
    RankedContext,
    SearchResult,
)


class PostExtractorPort(ABC):
    """Port for extracting structured content from a post URL."""

    @abstractmethod
    async def extract(self, url: str) -> PostContent:
        """Extract post content from a URL.

        Args:
            url: The post URL to extract.

        Returns:
            Structured PostContent.
        """
        ...


class SearchPort(ABC):
    """Port for searching the web for context relevant to a post."""

    @abstractmethod
    async def search(
        self, post: PostContent, queries: list[str] | None = None
    ) -> list[SearchResult]:
        """Search for context relevant to a post.

        Args:
            post: The extracted post content.
            queries: Optional pre-computed search queries. If None, the
                adapter formulates queries from the post.

        Returns:
            List of search results.
        """
        ...


class RankerPort(ABC):
    """Port for reranking search results by relevance."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult]) -> list[RankedContext]:
        """Rerank search results by relevance to the query.

        Args:
            query: The original query or post text.
            results: Search results to rerank.

        Returns:
            Top-k results ranked by relevance.
        """
        ...


class LLMPort(ABC):
    """Port for LLM completion requests."""

    @abstractmethod
    async def completion(
        self, messages: list[dict], model: str | None = None, **kwargs
    ) -> dict:
        """Make an LLM completion request.

        Args:
            messages: Chat messages in OpenAI format.
            model: Optional specific model to use.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Dict with 'content', 'model', and 'usage' keys.
        """
        ...


class ExplanationPort(ABC):
    """Port for generating explanations from post content and context."""

    @abstractmethod
    async def generate(
        self,
        post: PostContent,
        contexts: list[RankedContext],
        images: list[ImageEmbed] | None = None,
        model: str | None = None,
    ) -> ExplanationResult:
        """Generate an explanation for a post given context.

        Args:
            post: The extracted post content.
            contexts: Ranked context snippets.
            images: Optional embedded images.
            model: Optional specific model to use.

        Returns:
            ExplanationResult with bullets, citations, model info.
        """
        ...
