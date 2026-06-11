"""FastAPI backend with Server-Sent Events streaming for the Contextual Post Explainer."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.adapters.bluesky_extractor import BlueskyExtractor
from src.adapters.explanation_generator import ExplanationGeneratorAdapter
from src.adapters.litellm_router import LiteLLMRouter
from src.adapters.tavily_searcher import TavilySearcher
from src.domain.entities import RankedContext, SearchResult
from src.domain.ports import RankerPort
from src.domain.use_cases import ExplainPostUseCase
from src.exceptions import ExplainerError

logger = logging.getLogger(__name__)


import os

def _create_ranker():
    """Create ranker — uses cross-encoder if ENABLE_CROSSENCODER=1, else score-based fallback."""
    if os.environ.get("ENABLE_CROSSENCODER") == "1":
        try:
            from src.adapters.crossencoder_reranker import CrossEncoderReranker
            return CrossEncoderReranker()
        except Exception as e:
            logger.warning("CrossEncoder failed to load: %s", e)

    class ScoreBasedReranker(RankerPort):
        """Fallback reranker using Tavily scores (no ML model needed)."""
        def rerank(self, query: str, results: list[SearchResult]) -> list[RankedContext]:
            ranked = [
                RankedContext(title=r.title, url=r.url, content=r.content, relevance_score=r.score)
                for r in results
            ]
            ranked.sort(key=lambda x: x.relevance_score, reverse=True)
            return ranked[:5]

    return ScoreBasedReranker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize heavy resources once."""
    app.state.ranker = _create_ranker()
    yield


app = FastAPI(
    title="Contextual Post Explainer",
    description="Explains Bluesky posts by retrieving relevant web context",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApiKeys(BaseModel):
    """API keys provided by the user per-request."""
    anthropic: str | None = None
    openai: str | None = None
    tavily: str | None = None


class ExplainRequest(BaseModel):
    """Request body for the explain endpoint."""

    url: str
    provider: str | None = None
    api_keys: ApiKeys | None = None


def _resolve_provider(api_keys: ApiKeys | None) -> str | None:
    """Auto-detect provider from which API key was provided."""
    if not api_keys:
        return None
    if api_keys.anthropic:
        return "anthropic/claude-sonnet-4-20250514"
    if api_keys.openai:
        return "openai/gpt-4o"
    return None


def _apply_api_keys(keys: ApiKeys | None) -> None:
    """Set API keys as environment variables for the current request."""
    if not keys:
        return
    if keys.anthropic:
        os.environ["ANTHROPIC_API_KEY"] = keys.anthropic
    if keys.openai:
        os.environ["OPENAI_API_KEY"] = keys.openai
    if keys.tavily:
        os.environ["TAVILY_API_KEY"] = keys.tavily


def _build_use_case(ranker, provider: str | None = None, tavily_key: str | None = None) -> ExplainPostUseCase:
    """Build use case with all adapters."""
    extractor = BlueskyExtractor()
    searcher = TavilySearcher(api_key=tavily_key)
    llm = LiteLLMRouter()
    explainer = ExplanationGeneratorAdapter(llm=llm)
    return ExplainPostUseCase(
        extractor=extractor,
        searcher=searcher,
        ranker=ranker,
        explainer=explainer,
        provider=provider,
        verbose=True,
    )


async def _sse_generator(url: str, provider: str | None, ranker, api_keys: ApiKeys | None):
    """Async generator yielding SSE events as the pipeline progresses."""
    # Apply user-provided API keys to environment
    _apply_api_keys(api_keys)

    tavily_key = api_keys.tavily if api_keys else None
    use_case = _build_use_case(ranker, provider, tavily_key=tavily_key)

    try:
        # Step 1: Extract post
        yield f"event: status\ndata: {json.dumps({'step': 'extracting', 'message': 'Extracting post content...'})}\n\n"
        post = await use_case.extractor.extract(url)
        yield f"event: post\ndata: {json.dumps({'author': post.author_handle, 'text': post.text[:200], 'hashtags': post.hashtags})}\n\n"

        # Step 2: Search
        yield f"event: status\ndata: {json.dumps({'step': 'searching', 'message': 'Searching for context...'})}\n\n"
        results = await use_case.searcher.search(post)
        sources = [{"title": r.title, "url": r.url} for r in results[:5]]
        yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"

        # Step 3: Rerank
        yield f"event: status\ndata: {json.dumps({'step': 'reranking', 'message': 'Reranking results...'})}\n\n"
        ranked = ranker.rerank(post.text, results)

        # Step 4: Generate explanation
        yield f"event: status\ndata: {json.dumps({'step': 'generating', 'message': 'Generating explanation...'})}\n\n"
        explanation = await use_case.explainer.generate(
            post,
            ranked,
            images=post.images if post.images else None,
            model=provider,
        )

        # Stream bullets one by one
        for bullet in explanation.bullets:
            yield f"event: bullet\ndata: {json.dumps({'bullet': bullet})}\n\n"
            await asyncio.sleep(0.1)  # Small delay for visual effect

        # Final metadata
        yield f"event: done\ndata: {json.dumps({'model_used': explanation.model_used, 'citations': explanation.citations})}\n\n"

    except ValueError as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'type': 'validation'})}\n\n"
    except ExplainerError as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'type': 'explainer'})}\n\n"
    except Exception as e:
        logger.exception("Unexpected error in SSE stream")
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'type': 'internal'})}\n\n"


@app.post("/api/explain")
async def explain_post(request: ExplainRequest):
    """Start explanation pipeline and stream results via SSE."""
    # Auto-detect provider from keys if not explicitly set
    provider = request.provider or _resolve_provider(request.api_keys)
    return StreamingResponse(
        _sse_generator(request.url, provider, app.state.ranker, request.api_keys),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
