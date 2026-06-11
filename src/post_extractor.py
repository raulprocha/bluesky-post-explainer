"""Post Extractor module for the Contextual Post Explainer agent.

Handles parsing Bluesky URLs, resolving handles to DIDs,
fetching post threads via the AT Protocol, and extracting structured content.
"""

import asyncio
import re
from dataclasses import dataclass, field

import httpx

from src.exceptions import NetworkError, PostBlockedError, PostNotFoundError

# Regex for Bluesky post URLs
_BSKY_URL_PATTERN = re.compile(
    r"^https://bsky\.app/profile/(?P<actor>[^/]+)/post/(?P<rkey>[a-zA-Z0-9]+)$"
)

# AT Protocol public API base
_PUBLIC_API_BASE = "https://public.api.bsky.app/xrpc"

# Retry configuration
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0


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


class PostExtractor:
    """Extracts content from Bluesky posts via the AT Protocol public API."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def parse_url(self, url: str) -> tuple[str, str]:
        """Extract (actor, rkey) from a Bluesky post URL.

        Args:
            url: A URL matching https://bsky.app/profile/{actor}/post/{rkey}

        Returns:
            Tuple of (actor_handle, record_key)

        Raises:
            ValueError: If the URL does not match the expected format.
        """
        match = _BSKY_URL_PATTERN.match(url)
        if not match:
            raise ValueError(
                f"Invalid Bluesky post URL: '{url}'. "
                f"Expected format: https://bsky.app/profile/{{actor}}/post/{{rkey}}"
            )
        return match.group("actor"), match.group("rkey")

    async def resolve_handle(self, handle: str) -> str:
        """Resolve a Bluesky handle to a DID.

        Args:
            handle: The Bluesky handle (e.g. 'user.bsky.social')

        Returns:
            The resolved DID string.

        Raises:
            PostNotFoundError: If the handle cannot be found (404).
            NetworkError: If the request fails after retries.
        """
        url = f"{_PUBLIC_API_BASE}/com.atproto.identity.resolveHandle"
        params = {"handle": handle}

        response = await self._request_with_retry("GET", url, params=params)

        if response.status_code == 400 or response.status_code == 404:
            raise PostNotFoundError(
                f"Handle '{handle}' could not be resolved. The account may not exist."
            )

        if response.status_code != 200:
            raise NetworkError(
                f"Failed to resolve handle '{handle}': HTTP {response.status_code}"
            )

        data = response.json()
        return data["did"]

    async def fetch_thread(self, did: str, rkey: str) -> dict:
        """Fetch a post thread from the AT Protocol API.

        Args:
            did: The author's DID.
            rkey: The post's record key.

        Returns:
            The thread data dictionary from the API response.

        Raises:
            NetworkError: If the request fails after retries.
        """
        at_uri = f"at://{did}/app.bsky.feed.post/{rkey}"
        url = f"{_PUBLIC_API_BASE}/app.bsky.feed.getPostThread"
        params = {"uri": at_uri, "depth": "0", "parentHeight": "0"}

        response = await self._request_with_retry("GET", url, params=params)

        if response.status_code != 200:
            raise NetworkError(
                f"Failed to fetch post thread: HTTP {response.status_code} "
                f"from {_PUBLIC_API_BASE}"
            )

        return response.json()

    def parse_thread(self, thread_data: dict) -> PostContent:
        """Parse thread response JSON into a PostContent object.

        Args:
            thread_data: Raw JSON response from getPostThread.

        Returns:
            Structured PostContent with extracted fields.

        Raises:
            PostNotFoundError: If the thread indicates the post was not found.
            PostBlockedError: If the thread indicates the post is blocked.
        """
        thread = thread_data.get("thread", thread_data)

        # Check for error states
        thread_type = thread.get("$type", "")
        if thread_type == "app.bsky.feed.defs#notFoundPost":
            raise PostNotFoundError("Post not found.")
        if thread_type == "app.bsky.feed.defs#blockedPost":
            raise PostBlockedError("Post is blocked due to moderation.")

        # Extract post data from threadViewPost
        post = thread.get("post", {})
        record = post.get("record", {})
        author = post.get("author", {})

        text = record.get("text", "")
        author_handle = author.get("handle", "")
        author_did = author.get("did", "")
        timestamp = record.get("createdAt", "")

        # Parse facets
        hashtags: list[str] = []
        mentions: list[str] = []
        links: list[str] = []

        facets = record.get("facets", [])
        for facet in facets:
            features = facet.get("features", [])
            for feature in features:
                feat_type = feature.get("$type", "")
                if feat_type == "app.bsky.richtext.facet#tag":
                    tag = feature.get("tag", "")
                    if tag:
                        hashtags.append(tag)
                elif feat_type == "app.bsky.richtext.facet#mention":
                    did = feature.get("did", "")
                    if did:
                        mentions.append(did)
                elif feat_type == "app.bsky.richtext.facet#link":
                    uri = feature.get("uri", "")
                    if uri:
                        links.append(uri)

        # Parse image embeds
        images: list[ImageEmbed] = []
        embed = post.get("embed", {})
        if embed:
            embed_type = embed.get("$type", "")
            if embed_type == "app.bsky.embed.images#view":
                for img in embed.get("images", []):
                    images.append(
                        ImageEmbed(
                            url=img.get("fullsize", ""),
                            alt_text=img.get("alt", ""),
                            mime_type=img.get("mimeType", "image/jpeg"),
                        )
                    )

        # Extract rkey from the post URI
        rkey = ""
        post_uri = post.get("uri", "")
        if post_uri:
            # AT URI format: at://did/app.bsky.feed.post/rkey
            parts = post_uri.rsplit("/", 1)
            if len(parts) == 2:
                rkey = parts[1]

        return PostContent(
            text=text,
            author_handle=author_handle,
            author_did=author_did,
            timestamp=timestamp,
            hashtags=hashtags,
            mentions=mentions,
            links=links,
            images=images,
            rkey=rkey,
        )

    async def extract(self, url: str) -> PostContent:
        """Full extraction pipeline: parse URL → resolve handle → fetch thread → parse.

        Args:
            url: A Bluesky post URL.

        Returns:
            Structured PostContent with all extracted data.

        Raises:
            ValueError: If the URL format is invalid.
            PostNotFoundError: If the post or handle doesn't exist.
            PostBlockedError: If the post is blocked.
            NetworkError: If network requests fail after retries.
        """
        actor, rkey = self.parse_url(url)

        # If actor looks like a DID already, skip resolution
        if actor.startswith("did:"):
            did = actor
        else:
            did = await self.resolve_handle(actor)

        thread_data = await self.fetch_thread(did, rkey)
        post_content = self.parse_thread(thread_data)

        # Ensure rkey is set even if not present in the thread response
        if not post_content.rkey:
            post_content.rkey = rkey

        return post_content

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: dict | None = None,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff retry for transient errors.

        Retries on timeouts and 5xx status codes up to _MAX_RETRIES times.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL.
            params: Optional query parameters.

        Returns:
            The HTTP response.

        Raises:
            NetworkError: If all retries are exhausted due to transient errors.
        """
        client = await self._get_client()
        last_exception: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.request(method, url, params=params)

                # Don't retry client errors (4xx) except for rate limits
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < _MAX_RETRIES:
                        backoff = _BASE_BACKOFF_SECONDS * (2**attempt)
                        await asyncio.sleep(backoff)
                        continue

                return response

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < _MAX_RETRIES:
                    backoff = _BASE_BACKOFF_SECONDS * (2**attempt)
                    await asyncio.sleep(backoff)
                    continue

        # All retries exhausted
        raise NetworkError(
            f"Request to {url} failed after {_MAX_RETRIES + 1} attempts: "
            f"{last_exception}"
        )
