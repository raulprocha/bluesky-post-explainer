"""Core use cases for the Contextual Post Explainer.

Contains the application logic orchestrating domain ports,
independent of any infrastructure or framework concerns.
"""

import logging

from src.domain.entities import (
    ExplanationResult,
    PostContent,
    RankedContext,
    SearchResult,
)
from src.domain.ports import ExplanationPort, PostExtractorPort, RankerPort, SearchPort

logger = logging.getLogger(__name__)


class ExplainPostUseCase:
    """Orchestrates the full post explanation pipeline.

    Coordinates extraction, search, reranking, and generation
    through port interfaces, remaining decoupled from concrete adapters.
    """

    def __init__(
        self,
        extractor: PostExtractorPort,
        searcher: SearchPort,
        ranker: RankerPort,
        explainer: ExplanationPort,
        provider: str | None = None,
        verbose: bool = False,
    ):
        """Initialize with port implementations.

        Args:
            extractor: Post extraction port.
            searcher: Search port.
            ranker: Reranking port.
            explainer: Explanation generation port.
            provider: Optional LLM provider/model name.
            verbose: If True, store intermediate pipeline data.
        """
        self.extractor = extractor
        self.searcher = searcher
        self.ranker = ranker
        self.explainer = explainer
        self.provider = provider
        self.verbose = verbose

        # Verbose intermediate data
        self.last_post_content: PostContent | None = None
        self.last_search_results: list[SearchResult] | None = None
        self.last_ranked_contexts: list[RankedContext] | None = None

    async def execute(self, url: str) -> ExplanationResult:
        """Run the full explanation pipeline for a post URL.

        Pipeline steps:
            1. Extract post content from URL
            2. Search for relevant web context
            3. Rerank results using cross-encoder
            4. Generate explanatory bullets via LLM

        Args:
            url: A valid Bluesky post URL.

        Returns:
            ExplanationResult with 3-5 grounded bullets, citations,
            and model info.
        """
        # Step 1: Extract post content
        logger.info("Extracting post content from %s", url)
        post = await self.extractor.extract(url)
        if self.verbose:
            self.last_post_content = post

        # Step 2: Search for relevant context
        logger.info("Searching for relevant context...")
        results = await self.searcher.search(post)
        if self.verbose:
            self.last_search_results = results

        # Step 3: Rerank results by relevance
        logger.info("Reranking %d results...", len(results))
        ranked = self.ranker.rerank(post.text, results)
        if self.verbose:
            self.last_ranked_contexts = ranked

        # Step 4: Generate explanation
        logger.info("Generating explanation...")
        explanation = await self.explainer.generate(
            post,
            ranked,
            images=post.images if post.images else None,
            model=self.provider,
        )

        return explanation
