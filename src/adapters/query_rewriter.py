"""LLM-based query rewriter.

Transforms a conversational social media post into focused web search
queries, improving retrieval quality over using the raw post text.
"""

import logging

from src.domain.entities import PostContent
from src.domain.ports import LLMPort

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """You are a search query optimizer. Given a social media post, \
produce 1-3 concise web search queries that would find background context to \
explain the post to someone unfamiliar with it.

Rules:
- Extract key entities (people, organizations, events, products) and topics.
- Each query should be 3-10 words — concise, not a full sentence.
- Focus on what a reader would need to look up to understand the post.
- Output ONLY the queries, one per line. No numbering, no extra text.

Post by @{author}:
{text}{hashtags}"""


class QueryRewriter:
    """Rewrites posts into focused search queries using an LLM."""

    def __init__(self, llm: LLMPort, model: str | None = None):
        """Initialize with an LLM port.

        Args:
            llm: LLM port used to generate queries.
            model: Optional specific model identifier.
        """
        self._llm = llm
        self._model = model

    async def rewrite(self, post: PostContent) -> list[str]:
        """Generate focused search queries from a post.

        Args:
            post: The extracted post content.

        Returns:
            List of 1-3 focused query strings. Falls back to the raw post
            text (truncated) if the LLM call fails or returns nothing.
        """
        hashtags = ""
        if post.hashtags:
            hashtags = "\nHashtags: " + ", ".join(f"#{h}" for h in post.hashtags)

        prompt = _REWRITE_PROMPT.format(
            author=post.author_handle,
            text=post.text,
            hashtags=hashtags,
        )

        try:
            response = await self._llm.completion(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
            )
            content = response["content"]
            queries = [
                line.strip(" -•\t")
                for line in content.strip().split("\n")
                if line.strip()
            ]
            queries = [q for q in queries if q][:3]
            if queries:
                return queries
        except Exception as e:
            logger.warning("Query rewriting failed, falling back to raw text: %s", e)

        # Fallback: truncated raw post text
        fallback = post.text.strip()[:400]
        return [fallback] if fallback else []
