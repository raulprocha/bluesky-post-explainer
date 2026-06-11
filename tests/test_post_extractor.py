"""Unit tests for the Post Extractor module."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from src.adapters.bluesky_extractor import BlueskyExtractor
from src.domain.entities import PostContent, ImageEmbed


class TestParseUrl:
    """Tests for URL parsing."""

    def setup_method(self):
        self.extractor = BlueskyExtractor()

    def test_valid_url_simple_handle(self):
        actor, rkey = self.extractor.parse_url(
            "https://bsky.app/profile/user.bsky.social/post/abc123"
        )
        assert actor == "user.bsky.social"
        assert rkey == "abc123"

    def test_valid_url_custom_domain(self):
        actor, rkey = self.extractor.parse_url(
            "https://bsky.app/profile/jay.bsky.team/post/3k2a5b7c9d0"
        )
        assert actor == "jay.bsky.team"
        assert rkey == "3k2a5b7c9d0"

    def test_valid_url_did_as_actor(self):
        actor, rkey = self.extractor.parse_url(
            "https://bsky.app/profile/did:plc:abcdef123456/post/xyz789"
        )
        assert actor == "did:plc:abcdef123456"
        assert rkey == "xyz789"

    def test_invalid_url_wrong_domain(self):
        with pytest.raises(ValueError, match="Invalid Bluesky post URL"):
            self.extractor.parse_url(
                "https://twitter.com/profile/user/post/abc123"
            )

    def test_invalid_url_missing_post_segment(self):
        with pytest.raises(ValueError, match="Invalid Bluesky post URL"):
            self.extractor.parse_url(
                "https://bsky.app/profile/user.bsky.social"
            )

    def test_invalid_url_empty_string(self):
        with pytest.raises(ValueError, match="Invalid Bluesky post URL"):
            self.extractor.parse_url("")

    def test_invalid_url_random_string(self):
        with pytest.raises(ValueError, match="Invalid Bluesky post URL"):
            self.extractor.parse_url("not a url at all")

    def test_invalid_url_trailing_slash(self):
        with pytest.raises(ValueError, match="Invalid Bluesky post URL"):
            self.extractor.parse_url(
                "https://bsky.app/profile/user.bsky.social/post/abc123/"
            )


class TestParseThread:
    """Tests for thread JSON parsing."""

    def setup_method(self):
        self.extractor = BlueskyExtractor()

    def test_basic_thread(self):
        thread_data = {
            "thread": {
                "$type": "app.bsky.feed.defs#threadViewPost",
                "post": {
                    "uri": "at://did:plc:abc123/app.bsky.feed.post/rkey456",
                    "author": {
                        "handle": "alice.bsky.social",
                        "did": "did:plc:abc123",
                    },
                    "record": {
                        "text": "Hello, world!",
                        "createdAt": "2024-01-15T10:30:00.000Z",
                        "facets": [],
                    },
                    "embed": {},
                },
            }
        }

        result = self.extractor.parse_thread(thread_data)
        assert result.text == "Hello, world!"
        assert result.author_handle == "alice.bsky.social"
        assert result.author_did == "did:plc:abc123"
        assert result.timestamp == "2024-01-15T10:30:00.000Z"
        assert result.rkey == "rkey456"
        assert result.hashtags == []
        assert result.mentions == []
        assert result.links == []
        assert result.images == []

    def test_thread_with_facets(self):
        thread_data = {
            "thread": {
                "$type": "app.bsky.feed.defs#threadViewPost",
                "post": {
                    "uri": "at://did:plc:abc/app.bsky.feed.post/rk1",
                    "author": {"handle": "bob.bsky.social", "did": "did:plc:bob"},
                    "record": {
                        "text": "Check out #python and @alice.bsky.social and https://example.com",
                        "createdAt": "2024-02-01T12:00:00.000Z",
                        "facets": [
                            {
                                "index": {"byteStart": 10, "byteEnd": 17},
                                "features": [
                                    {
                                        "$type": "app.bsky.richtext.facet#tag",
                                        "tag": "python",
                                    }
                                ],
                            },
                            {
                                "index": {"byteStart": 22, "byteEnd": 42},
                                "features": [
                                    {
                                        "$type": "app.bsky.richtext.facet#mention",
                                        "did": "did:plc:alice",
                                    }
                                ],
                            },
                            {
                                "index": {"byteStart": 47, "byteEnd": 66},
                                "features": [
                                    {
                                        "$type": "app.bsky.richtext.facet#link",
                                        "uri": "https://example.com",
                                    }
                                ],
                            },
                        ],
                    },
                    "embed": {},
                },
            }
        }

        result = self.extractor.parse_thread(thread_data)
        assert result.hashtags == ["python"]
        assert result.mentions == ["did:plc:alice"]
        assert result.links == ["https://example.com"]

    def test_thread_with_images(self):
        thread_data = {
            "thread": {
                "$type": "app.bsky.feed.defs#threadViewPost",
                "post": {
                    "uri": "at://did:plc:abc/app.bsky.feed.post/img1",
                    "author": {"handle": "carol.bsky.social", "did": "did:plc:carol"},
                    "record": {
                        "text": "Look at this!",
                        "createdAt": "2024-03-01T08:00:00.000Z",
                        "facets": [],
                    },
                    "embed": {
                        "$type": "app.bsky.embed.images#view",
                        "images": [
                            {
                                "fullsize": "https://cdn.bsky.app/img/feed_fullsize/abc/123.jpg",
                                "alt": "A nice sunset",
                                "mimeType": "image/jpeg",
                            },
                            {
                                "fullsize": "https://cdn.bsky.app/img/feed_fullsize/abc/456.png",
                                "alt": "",
                                "mimeType": "image/png",
                            },
                        ],
                    },
                },
            }
        }

        result = self.extractor.parse_thread(thread_data)
        assert len(result.images) == 2
        assert result.images[0].url == "https://cdn.bsky.app/img/feed_fullsize/abc/123.jpg"
        assert result.images[0].alt_text == "A nice sunset"
        assert result.images[0].mime_type == "image/jpeg"
        assert result.images[1].url == "https://cdn.bsky.app/img/feed_fullsize/abc/456.png"
        assert result.images[1].alt_text == ""
        assert result.images[1].mime_type == "image/png"

    def test_not_found_post(self):
        thread_data = {
            "thread": {
                "$type": "app.bsky.feed.defs#notFoundPost",
                "uri": "at://did:plc:abc/app.bsky.feed.post/missing",
                "notFound": True,
            }
        }

        from src.domain.exceptions import PostNotFoundError

        with pytest.raises(PostNotFoundError, match="not found"):
            self.extractor.parse_thread(thread_data)

    def test_blocked_post(self):
        thread_data = {
            "thread": {
                "$type": "app.bsky.feed.defs#blockedPost",
                "uri": "at://did:plc:abc/app.bsky.feed.post/blocked",
                "blocked": True,
            }
        }

        from src.domain.exceptions import PostBlockedError

        with pytest.raises(PostBlockedError, match="blocked"):
            self.extractor.parse_thread(thread_data)

    def test_post_with_no_text_only_image(self):
        """Edge case: post with no text, only an image embed."""
        thread_data = {
            "thread": {
                "$type": "app.bsky.feed.defs#threadViewPost",
                "post": {
                    "uri": "at://did:plc:abc/app.bsky.feed.post/notext",
                    "author": {"handle": "dave.bsky.social", "did": "did:plc:dave"},
                    "record": {
                        "text": "",
                        "createdAt": "2024-04-01T09:00:00.000Z",
                    },
                    "embed": {
                        "$type": "app.bsky.embed.images#view",
                        "images": [
                            {
                                "fullsize": "https://cdn.bsky.app/img/feed_fullsize/abc/solo.jpg",
                                "alt": "Just a picture",
                                "mimeType": "image/jpeg",
                            }
                        ],
                    },
                },
            }
        }

        result = self.extractor.parse_thread(thread_data)
        assert result.text == ""
        assert len(result.images) == 1
        assert result.images[0].alt_text == "Just a picture"

    def test_post_with_multiple_facets_same_type(self):
        """Post with multiple hashtags."""
        thread_data = {
            "thread": {
                "$type": "app.bsky.feed.defs#threadViewPost",
                "post": {
                    "uri": "at://did:plc:abc/app.bsky.feed.post/multi",
                    "author": {"handle": "eve.bsky.social", "did": "did:plc:eve"},
                    "record": {
                        "text": "#python #rust #go",
                        "createdAt": "2024-05-01T10:00:00.000Z",
                        "facets": [
                            {
                                "features": [
                                    {"$type": "app.bsky.richtext.facet#tag", "tag": "python"}
                                ]
                            },
                            {
                                "features": [
                                    {"$type": "app.bsky.richtext.facet#tag", "tag": "rust"}
                                ]
                            },
                            {
                                "features": [
                                    {"$type": "app.bsky.richtext.facet#tag", "tag": "go"}
                                ]
                            },
                        ],
                    },
                    "embed": {},
                },
            }
        }

        result = self.extractor.parse_thread(thread_data)
        assert result.hashtags == ["python", "rust", "go"]
