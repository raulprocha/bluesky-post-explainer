"""CLI entry point for the Contextual Post Explainer.

Instantiates adapters, wires the use case, and runs the pipeline
from the command line.
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from src.adapters.bluesky_extractor import BlueskyExtractor
from src.adapters.crossencoder_reranker import CrossEncoderReranker
from src.adapters.explanation_generator import ExplanationGeneratorAdapter
from src.adapters.litellm_router import LiteLLMRouter
from src.adapters.tavily_searcher import TavilySearcher
from src.domain.use_cases import ExplainPostUseCase
from src.domain.exceptions import ExplainerError


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
        help="LLM provider/model to use (e.g. claude-3-5-sonnet-20241022, gpt-4o)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Output intermediate steps before bullets",
    )
    return parser


def main() -> None:
    """Entry point: parse args, run use case, print bullets."""
    parser = _build_parser()
    args = parser.parse_args()

    # Wire adapters
    extractor = BlueskyExtractor()
    searcher = TavilySearcher()
    ranker = CrossEncoderReranker()
    llm = LiteLLMRouter()
    explainer = ExplanationGeneratorAdapter(llm=llm)

    use_case = ExplainPostUseCase(
        extractor=extractor,
        searcher=searcher,
        ranker=ranker,
        explainer=explainer,
        provider=args.provider,
        verbose=args.verbose,
    )

    try:
        result = asyncio.run(use_case.execute(args.url))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ExplainerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verbose output before bullets
    if args.verbose and use_case.last_post_content:
        post = use_case.last_post_content
        print(f"[Post] @{post.author_handle}: {post.text[:120]}")
        if use_case.last_search_results:
            print("[Search Results]")
            for r in use_case.last_search_results:
                print(f"  - {r.title} ({r.url})")
        if use_case.last_ranked_contexts:
            print("[Reranked]")
            for ctx in use_case.last_ranked_contexts:
                print(f"  [{ctx.relevance_score:.4f}] {ctx.title}")
        print()

    # Print explanatory bullets
    for bullet in result.bullets:
        print(f"• {bullet}")


if __name__ == "__main__":
    main()
