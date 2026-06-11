"""Agent Orchestrator for the Contextual Post Explainer.

Wires all pipeline components together and manages the full
extraction → search → rerank → generation flow.
"""

import logging

from src.context_retriever import ContextRetriever, SearchResult
from src.explanation_generator import ExplanationGenerator, ExplanationResult
from src.llm_router import LLMRouter
from src.post_extractor import PostContent, PostExtractor
from src.reranker import RankedContext, Reranker

logger = logging.getLogger(__name__)


class Agent:
    """Orchestrates the full post explanation pipeline.

    Coordinates PostExtractor, ContextRetriever, Reranker, and
    ExplanationGenerator to produce grounded explanatory bullets
    for a given Bluesky post URL.
    """

    def __init__(self, provider: str | None = None, verbose: bool = False):
        """Initialize all pipeline components.

        Args:
            provider: Optional LLM provider/model name to use for generation.
                If None, the LLMRouter will use its default fallback chain.
            verbose: If True, store intermediate pipeline data as attributes
                for inspection (last_post_content, last_search_results,
                last_ranked_contexts).
        """
        self.provider = provider
        self.verbose = verbose

        # Pipeline components
        self._post_extractor = PostExtractor()
        self._context_retriever = ContextRetriever()
        self._reranker = Reranker()
        self._llm_router = LLMRouter()
        self._explanation_generator = ExplanationGenerator(router=self._llm_router)

        # Verbose mode intermediate data (always initialized for attribute access)
        self.last_post_content: PostContent | None = None
        self.last_search_results: list[SearchResult] | None = None
        self.last_ranked_contexts: list[RankedContext] | None = None

    async def explain(self, url: str) -> ExplanationResult:
        """Run the full explanation pipeline for a Bluesky post URL.

        Pipeline steps:
            1. Extract post content from URL via AT Protocol
            2. Search for relevant web context via Tavily
            3. Rerank results using cross-encoder
            4. Generate explanatory bullets via LLM

        Args:
            url: A valid Bluesky post URL
                (https://bsky.app/profile/{actor}/post/{rkey}).

        Returns:
            ExplanationResult with 3-5 grounded bullets, citations,
            and model info.

        Raises:
            ValueError: If the URL format is invalid.
            PostNotFoundError: If the post doesn't exist.
            PostBlockedError: If the post is blocked.
            NetworkError: If network requests fail.
            SearchError: If Tavily API fails.
            AllProvidersFailedError: If all LLM providers fail.
            GenerationError: If the LLM output is malformed.
        """
        try:
            # Step 1: Extract post content
            logger.info("Extracting post content from %s", url)
            post_content = await self._post_extractor.extract(url)

            if self.verbose:
                self.last_post_content = post_content
                logger.info(
                    "Extracted post by @%s: %s",
                    post_content.author_handle,
                    post_content.text[:100],
                )

            # Step 2: Search for relevant context
            logger.info("Searching for relevant context...")
            search_results = await self._context_retriever.search(post_content)

            if self.verbose:
                self.last_search_results = search_results
                logger.info(
                    "Found %d search results", len(search_results)
                )

            # Step 3: Rerank results by relevance
            logger.info("Reranking %d results...", len(search_results))
            ranked_contexts = self._reranker.rerank(
                post_content.text, search_results
            )

            if self.verbose:
                self.last_ranked_contexts = ranked_contexts
                logger.info(
                    "Top %d contexts after reranking", len(ranked_contexts)
                )

            # Step 4: Generate explanation
            logger.info("Generating explanation...")
            result = await self._explanation_generator.generate(
                post=post_content,
                contexts=ranked_contexts,
                images=post_content.images if post_content.images else None,
                model=self.provider,
            )

            return result

        finally:
            # Always close the HTTP client to avoid resource leaks
            await self._post_extractor.close()
