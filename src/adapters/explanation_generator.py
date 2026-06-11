"""Explanation Generator adapter.

Implements ExplanationPort using an LLMPort to synthesize explanatory
bullets from post content, retrieved context, and optionally images.
"""

import base64
import logging
import re

import httpx

from src.domain.entities import (
    ExplanationResult,
    ImageEmbed,
    PostContent,
    RankedContext,
)
from src.domain.ports import ExplanationPort, LLMPort
from src.exceptions import GenerationError

logger = logging.getLogger(__name__)

# Regex to extract URLs from text
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\},\"']+")

# Regex to detect bullet lines
_BULLET_PATTERN = re.compile(r"^\s*(?:•|[-–—]|\d+[.\)])\s*(.+)", re.MULTILINE)

_SYSTEM_PROMPT = """You are an expert research assistant. Your job is to explain the context \
behind a social media post by producing exactly 3 to 5 explanatory bullet points.

Rules:
- Each bullet must be grounded in the provided source context. Do not invent facts.
- Include inline citations as URLs in parentheses when referencing a source.
- Be concise but informative. Each bullet should convey a distinct piece of context.
- Format each bullet on its own line starting with "• ".
- Output ONLY the bullet points, no preamble or conclusion."""


class ExplanationGeneratorAdapter(ExplanationPort):
    """Generates explanatory bullets from post content and retrieved context.

    Uses an LLMPort to synthesize 3-5 grounded bullets with
    inline citations. Supports multimodal input for posts with images.
    """

    def __init__(self, llm: LLMPort) -> None:
        """Initialize with an LLM port instance.

        Args:
            llm: An LLMPort implementation for making completion requests.
        """
        self._llm = llm

    def build_prompt(
        self,
        post: PostContent,
        contexts: list[RankedContext],
    ) -> list[dict]:
        """Build LLM message list with system and user prompts.

        The system message instructs the LLM to produce exactly 3-5 bullets.
        The user message includes the post content and retrieved context.

        Args:
            post: The extracted post content.
            contexts: Ranked context snippets with source URLs.

        Returns:
            List of message dicts in OpenAI chat format.
        """
        # Format context as numbered list
        context_block = ""
        for i, ctx in enumerate(contexts, 1):
            context_block += (
                f"{i}. [{ctx.title}]({ctx.url})\n"
                f"   {ctx.content[:500]}\n\n"
            )

        user_content = (
            f"## Post by @{post.author_handle}\n\n"
            f"{post.text}\n\n"
        )

        if post.hashtags:
            user_content += f"Hashtags: {', '.join('#' + h for h in post.hashtags)}\n\n"

        if post.mentions:
            user_content += f"Mentions: {', '.join(post.mentions)}\n\n"

        user_content += f"## Retrieved Context\n\n{context_block}"

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def process_images(self, images: list[ImageEmbed]) -> list[dict]:
        """Fetch and encode images as base64 for vision model input.

        For each ImageEmbed, fetches the image URL via httpx, converts to
        base64, and returns content blocks in OpenAI vision format.

        Args:
            images: List of ImageEmbed objects with URLs and mime types.

        Returns:
            List of content blocks in OpenAI image_url format.
        """
        image_blocks: list[dict] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for img in images:
                try:
                    response = await client.get(img.url)
                    response.raise_for_status()
                    b64_data = base64.b64encode(response.content).decode("utf-8")
                    mime_type = img.mime_type or "image/jpeg"
                    image_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_data}"
                            },
                        }
                    )
                except (httpx.HTTPError, httpx.TimeoutException) as e:
                    logger.warning("Failed to fetch image %s: %s", img.url, e)
                    # Skip images that fail to fetch rather than failing entirely

        return image_blocks

    async def generate(
        self,
        post: PostContent,
        contexts: list[RankedContext],
        images: list[ImageEmbed] | None = None,
        model: str | None = None,
    ) -> ExplanationResult:
        """Generate 3-5 explanatory bullets grounded in context.

        Builds the prompt, optionally processes images for vision models,
        calls the LLM via the port, and parses the response into bullets.

        Args:
            post: The extracted post content.
            contexts: Ranked context snippets with source URLs.
            images: Optional list of embedded images to include.
            model: Optional specific model to use.

        Returns:
            ExplanationResult with bullets, citations, and model info.

        Raises:
            GenerationError: If the LLM output cannot be parsed into 3-5 bullets.
        """
        messages = self.build_prompt(post, contexts)

        # Handle images for multimodal input
        if images:
            image_blocks = await self.process_images(images)
            if image_blocks:
                # Convert user message to multimodal format
                user_msg = messages[-1]
                text_content = user_msg["content"]
                # Build content array with text + images
                content_parts: list[dict] = [
                    {"type": "text", "text": text_content}
                ]
                content_parts.extend(image_blocks)
                messages[-1] = {"role": "user", "content": content_parts}

        # Call LLM via port
        response = await self._llm.completion(messages=messages, model=model)

        # Parse response into bullets
        content = response["content"]
        model_used = response["model"]

        bullets = self._parse_bullets(content)

        # Validate bullet count (must be 3-5)
        if len(bullets) < 3 or len(bullets) > 5:
            bullets = self._normalize_bullet_count(bullets, content)

        if len(bullets) < 3 or len(bullets) > 5:
            raise GenerationError(
                f"LLM produced {len(bullets)} bullets, expected 3-5. "
                f"Raw response: {content[:200]}"
            )

        # Extract citation URLs from bullets
        citations = self._extract_citations(bullets)

        return ExplanationResult(
            bullets=bullets,
            citations=citations,
            model_used=model_used,
        )

    def _parse_bullets(self, content: str) -> list[str]:
        """Parse LLM response content into individual bullet strings.

        Looks for lines starting with bullet markers (•, -, –, —, or numbered).

        Args:
            content: Raw LLM response text.

        Returns:
            List of bullet strings (without the bullet marker).
        """
        matches = _BULLET_PATTERN.findall(content)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # Fallback: split by newlines and filter non-empty lines
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        return lines

    def _normalize_bullet_count(
        self, bullets: list[str], raw_content: str
    ) -> list[str]:
        """Attempt to normalize bullet count to 3-5 range.

        If too many bullets, truncate to 5.
        If too few, try splitting the raw content differently.

        Args:
            bullets: Currently parsed bullets.
            raw_content: The raw LLM response text.

        Returns:
            Adjusted list of bullets, possibly still outside 3-5 range.
        """
        if len(bullets) > 5:
            return bullets[:5]

        if len(bullets) < 3 and len(bullets) > 0:
            # Try splitting by double newline
            parts = [p.strip() for p in raw_content.split("\n\n") if p.strip()]
            # Re-parse each part for bullet markers
            all_bullets: list[str] = []
            for part in parts:
                sub_matches = _BULLET_PATTERN.findall(part)
                if sub_matches:
                    all_bullets.extend(m.strip() for m in sub_matches if m.strip())
                elif part.strip():
                    all_bullets.append(part.strip())
            if 3 <= len(all_bullets) <= 5:
                return all_bullets

        return bullets

    def _extract_citations(self, bullets: list[str]) -> list[str]:
        """Extract unique URLs mentioned in bullet text.

        Args:
            bullets: List of bullet strings.

        Returns:
            Deduplicated list of URLs found across all bullets.
        """
        urls: list[str] = []
        seen: set[str] = set()

        for bullet in bullets:
            found = _URL_PATTERN.findall(bullet)
            for url in found:
                # Clean trailing punctuation that might have been captured
                url = url.rstrip(".,;:!?)")
                if url not in seen:
                    seen.add(url)
                    urls.append(url)

        return urls
