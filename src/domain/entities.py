"""Pure domain entities for the Contextual Post Explainer.

These dataclasses have zero external dependencies and represent
the core data structures shared across all layers.
"""

from dataclasses import dataclass, field


@dataclass
class ImageEmbed:
    """Represents an embedded image in a Bluesky post."""

    url: str
    alt_text: str
    mime_type: str


@dataclass
class PostContent:
    """Structured content extracted from a Bluesky post."""

    text: str
    author_handle: str
    author_did: str
    timestamp: str
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    images: list[ImageEmbed] = field(default_factory=list)
    rkey: str = ""


@dataclass
class SearchResult:
    """A single search result from a search provider."""

    title: str
    url: str
    content: str
    score: float


@dataclass
class RankedContext:
    """A search result scored and ranked by relevance to the query."""

    title: str
    url: str
    content: str
    relevance_score: float


@dataclass
class ExplanationResult:
    """Result of explanation generation."""

    bullets: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    model_used: str = ""
