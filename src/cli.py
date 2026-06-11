"""CLI entry point for the Contextual Post Explainer agent.

Provides a simple command-line interface to invoke the agent with a
Bluesky post URL and optional flags for provider selection and verbose
output.
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from src.agent import Agent
from src.exceptions import ExplainerError


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="explain",
        description="Explain a Bluesky post by retrieving relevant web context.",
    )
    parser.add_argument(
        "url",
        help="Bluesky post URL (https://bsky.app/profile/{actor}/post/{rkey})",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default=None,
        help="LLM provider/model to use for generation (e.g. claude-3-5-sonnet-20241022, gpt-4o)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Output intermediate steps (search queries, source URLs, reranking scores) before bullets",
    )
    return parser


def _print_verbose(agent: Agent) -> None:
    """Print intermediate pipeline data when verbose mode is active."""
    # Search queries (from post content used for retrieval)
    if agent.last_post_content:
        post = agent.last_post_content
        print(f"[Post] @{post.author_handle}: {post.text[:120]}")
        if post.hashtags:
            print(f"[Hashtags] {', '.join(post.hashtags)}")
        print()

    # Source URLs from search results
    if agent.last_search_results:
        print("[Search Results]")
        for result in agent.last_search_results:
            print(f"  - {result.title} ({result.url})")
        print()

    # Reranking scores
    if agent.last_ranked_contexts:
        print("[Reranked Contexts]")
        for ctx in agent.last_ranked_contexts:
            print(f"  [{ctx.relevance_score:.4f}] {ctx.title} ({ctx.url})")
        print()


def main() -> None:
    """Entry point: parse args, run agent, print bullets."""
    parser = _build_parser()
    args = parser.parse_args()

    agent = Agent(provider=args.provider, verbose=args.verbose)

    try:
        result = asyncio.run(agent.explain(args.url))
    except ValueError as exc:
        print(
            f"Error: {exc}\n"
            f"Expected format: https://bsky.app/profile/{{actor}}/post/{{rkey}}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ExplainerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verbose output before bullets
    if args.verbose:
        _print_verbose(agent)

    # Print explanatory bullets
    for bullet in result.bullets:
        print(f"• {bullet}")


if __name__ == "__main__":
    main()
