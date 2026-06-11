"""Local (CPU) LLM-based query rewriter.

Uses a small instruction-tuned model (Qwen2.5-0.5B-Instruct) running on CPU
to rewrite posts into focused search queries — no external API tokens needed.

Trade-off vs. the API-based QueryRewriter: lower quality and higher latency
(model inference on CPU), but zero API cost. See README for comparative metrics.
"""

import logging

from src.domain.entities import PostContent

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_SYSTEM_PROMPT = (
    "You convert social media posts into a short web search query (3-8 words) "
    "to find background context. Output ONLY the query, nothing else."
)


class LocalQueryRewriter:
    """Rewrites posts into search queries using a small local CPU model.

    The model is lazily loaded on first use to avoid startup cost when the
    rewriter is not exercised.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        """Initialize the rewriter.

        Args:
            model_name: HuggingFace model identifier for a causal instruct model.
        """
        self._model_name = model_name
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazily load the tokenizer and model on first use."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local query rewriter model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)

    async def rewrite(self, post: PostContent) -> list[str]:
        """Generate focused search queries from a post.

        Args:
            post: The extracted post content.

        Returns:
            List of 1-2 query strings. Adds hashtags as a secondary query when
            present. Falls back to truncated raw text on failure.
        """
        try:
            self._ensure_loaded()
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": post.text},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self._tokenizer(text, return_tensors="pt").input_ids
            output = self._model.generate(
                input_ids, max_new_tokens=25, do_sample=False
            )
            query = self._tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            queries: list[str] = []
            if query:
                queries.append(query)
            if post.hashtags:
                hashtag_query = " ".join(post.hashtags)
                if hashtag_query and hashtag_query not in queries:
                    queries.append(hashtag_query)

            if queries:
                return queries[:2]
        except Exception as e:
            logger.warning(
                "Local query rewriting failed, falling back to raw text: %s", e
            )

        fallback = post.text.strip()[:400]
        return [fallback] if fallback else []
