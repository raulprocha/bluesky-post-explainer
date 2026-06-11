"""Reranker module for the Contextual Post Explainer agent.

Reranks search results by relevance to the original query using a
cross-encoder model (sentence-transformers CrossEncoder).
"""

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

from src.context_retriever import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankedContext:
    """A search result scored and ranked by relevance to the query."""

    title: str
    url: str
    content: str
    relevance_score: float


class Reranker:
    """Reranks search results using a cross-encoder model.

    Uses sentence-transformers CrossEncoder to score (query, document)
    pairs and returns the top-k most relevant results sorted by score
    in strictly non-increasing order.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
            top_k: Maximum number of results to return after reranking.

        Raises:
            RuntimeError: If the model fails to load.
        """
        self._top_k = top_k
        try:
            self._model = CrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}': {e}"
            ) from e

    def rerank(self, query: str, results: list[SearchResult]) -> list[RankedContext]:
        """Score and sort results by relevance to the query.

        Scores all (query, result.content) pairs using the cross-encoder,
        then returns the top-k results sorted in non-increasing order by
        relevance_score.

        Args:
            query: The original search query or post text.
            results: List of search results to rerank.

        Returns:
            Top-k RankedContext objects sorted by relevance_score descending.
            Returns empty list if input is empty.
            Returns all results sorted if k > len(results).
        """
        if not results:
            return []

        # Build (query, document) pairs for batch scoring
        pairs = [(query, result.content) for result in results]

        # Batch score all pairs at once
        scores = self._model.predict(pairs)

        # Ensure scores is a numpy array for consistent handling
        scores = np.array(scores, dtype=np.float64)

        # Create RankedContext objects with scores
        ranked: list[RankedContext] = []
        for result, score in zip(results, scores):
            ranked.append(
                RankedContext(
                    title=result.title,
                    url=result.url,
                    content=result.content,
                    relevance_score=float(score),
                )
            )

        # Sort by relevance_score in non-increasing order
        ranked.sort(key=lambda ctx: ctx.relevance_score, reverse=True)

        # Return top-k (or all if fewer than k)
        return ranked[: self._top_k]
