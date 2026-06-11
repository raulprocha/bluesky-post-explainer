"""Tavily Search adapter.

Implements SearchPort using the Tavily Search API to retrieve
relevant web context for a given post.
"""

import asyncio
import logging
import os

from tavily import TavilyClient

from src.domain.entities import PostContent, SearchResult
from src.domain.ports import SearchPort
from src.domain.exceptions import SearchError

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0

# Query constraints
_MAX_QUERIES = 3
_MAX_QUERY_LENGTH = 400  # Tavily query length limit
_MAX_RESULTS_PER_QUERY = 5


class TavilySearcher(SearchPort):
    """Retrieves web context for Bluesky posts using Tavily Search API."""

    def __init__(self, api_key: str | None = None, topic: str = "general"):
        """Initialize with Tavily API key.

        Args:
            api_key: Tavily API key. If None, reads from TAVILY_API_KEY env var.
            topic: Tavily search topic. "general" covers both news and
                encyclopedic context; "news" restricts to journalistic sources.

        Raises:
            SearchError: If no API key is available.
        """
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise SearchError(
                "Tavily API key not found. Set TAVILY_API_KEY environment variable."
            )
        self._client = TavilyClient(api_key=key)
        self._topic = topic

    def formulate_queries(self, post: PostContent) -> list[str]:
        """Generate 1-3 search queries from post text, hashtags, and mentions.

        Strategy:
        - Primary query: the full post text (truncated if needed)
        - Secondary query: focused query from hashtags (if present)
        - Maximum 3 queries total

        Args:
            post: Extracted post content.

        Returns:
            List of 1-3 non-empty query strings.
        """
        queries: list[str] = []

        # Primary query: use the post text directly (truncated if too long)
        if post.text:
            primary = post.text.strip()
            if len(primary) > _MAX_QUERY_LENGTH:
                primary = primary[:_MAX_QUERY_LENGTH].rsplit(" ", 1)[0]
            if primary:
                queries.append(primary)

        # Secondary query: hashtags combined into a focused query
        if post.hashtags and len(queries) < _MAX_QUERIES:
            hashtag_query = " ".join(f"#{tag}" for tag in post.hashtags)
            if len(hashtag_query) > _MAX_QUERY_LENGTH:
                hashtag_query = hashtag_query[:_MAX_QUERY_LENGTH].rsplit(" ", 1)[0]
            if hashtag_query and hashtag_query not in queries:
                queries.append(hashtag_query)

        # Tertiary query: if we have mentions, create a query about the mentioned entities
        if post.mentions and len(queries) < _MAX_QUERIES:
            text_snippet = post.text[:100].strip() if post.text else ""
            if text_snippet:
                mention_query = f"{text_snippet} context"
                if len(mention_query) > _MAX_QUERY_LENGTH:
                    mention_query = mention_query[:_MAX_QUERY_LENGTH].rsplit(" ", 1)[0]
                if mention_query and mention_query not in queries:
                    queries.append(mention_query)

        # Fallback
        if not queries:
            if post.hashtags:
                queries.append(" ".join(post.hashtags))
            elif post.links:
                queries.append(post.links[0])

        return queries[:_MAX_QUERIES]

    async def execute_search(self, query: str) -> list[SearchResult]:
        """Execute a single query against Tavily API with retry.

        Args:
            query: The search query string.

        Returns:
            List of SearchResult objects from Tavily.

        Raises:
            SearchError: If the API call fails after all retries.
        """
        last_exception: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                # TavilyClient.search is synchronous, wrap in asyncio.to_thread
                response = await asyncio.to_thread(
                    self._client.search,
                    query=query,
                    topic=self._topic,
                    max_results=_MAX_RESULTS_PER_QUERY,
                )

                results: list[SearchResult] = []
                for item in response.get("results", []):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            content=item.get("content", ""),
                            score=item.get("score", 0.0),
                        )
                    )
                return results

            except Exception as e:
                last_exception = e
                if attempt < _MAX_RETRIES:
                    backoff = _BASE_BACKOFF_SECONDS * (2**attempt)
                    logger.warning(
                        "Tavily API call failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        str(e),
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

        # All retries exhausted
        raise SearchError(
            f"Tavily search failed after {_MAX_RETRIES + 1} attempts: {last_exception}"
        )

    async def search(
        self, post: PostContent, queries: list[str] | None = None
    ) -> list[SearchResult]:
        """Formulate queries and execute searches, aggregating results.

        Deduplicates results by URL across multiple queries.

        Args:
            post: Extracted post content.
            queries: Optional pre-computed search queries (e.g. from an LLM
                query rewriter). If None, queries are formulated from the post.

        Returns:
            Aggregated and deduplicated list of SearchResult objects.

        Raises:
            SearchError: If Tavily API fails after retries for any query.
        """
        queries = queries or self.formulate_queries(post)

        if not queries:
            logger.warning("No queries formulated for post. Returning empty results.")
            return []

        # Execute all queries
        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            results = await self.execute_search(query)
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)

        if not all_results:
            logger.warning(
                "Tavily returned zero results for all %d queries.", len(queries)
            )

        return all_results
