"""Evaluation Harness for the Contextual Post Explainer.

Orchestrates running the agent on curated test cases, scoring outputs
via LLM-as-judge (RAG Triad), and generating aggregate reports.

Requirements: 6.6 [Design: Eval Harness]
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.domain.use_cases import ExplainPostUseCase
from src.domain.entities import ExplanationResult, RankedContext, SearchResult
from src.domain.ports import RankerPort
from src.adapters.bluesky_extractor import BlueskyExtractor
from src.adapters.crossencoder_reranker import CrossEncoderReranker
from src.adapters.explanation_generator import ExplanationGeneratorAdapter
from src.adapters.litellm_router import LiteLLMRouter as LLMRouter
from src.adapters.tavily_searcher import TavilySearcher
from eval.judge import score_faithfulness, score_relevance, score_helpfulness

logger = logging.getLogger(__name__)

# Default path to test cases file relative to project root
DEFAULT_TEST_CASES_PATH = "eval/test_cases.json"


class _ScoreReranker(RankerPort):
    """Lightweight reranker fallback using search provider scores.

    Used when the cross-encoder model (PyTorch) is not enabled, avoiding
    heavy ML dependencies during evaluation runs.
    """

    def rerank(self, query: str, results: list[SearchResult]) -> list[RankedContext]:
        ranked = [
            RankedContext(
                title=r.title, url=r.url, content=r.content, relevance_score=r.score
            )
            for r in results
        ]
        ranked.sort(key=lambda x: x.relevance_score, reverse=True)
        return ranked[:5]


@dataclass
class TestCase:
    """A single evaluation test case."""

    post_url: str
    description: str
    expected_themes: list[str]


@dataclass
class EvalScore:
    """Scores for a single evaluation across the RAG Triad dimensions."""

    faithfulness: float  # 0.0-1.0
    answer_relevance: float  # 1.0-5.0
    context_helpfulness: float  # 1.0-5.0


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    test_case: TestCase
    explanation: ExplanationResult | None
    scores: EvalScore | None
    reasoning: dict[str, str] = field(default_factory=dict)
    error: str | None = None


class EvalHarness:
    """Orchestrates evaluation of agent outputs using LLM-as-judge scoring.

    Runs the agent on test cases, scores each output on three dimensions
    (Faithfulness, Answer Relevance, Context Helpfulness), and generates
    aggregate reports with per-case scores and overall averages.
    """

    def __init__(self, provider: str | None = None) -> None:
        """Initialize with optional provider for both agent and judge.

        Args:
            provider: Optional LLM provider/model name. Used for both the
                agent's explanation generation and the judge's scoring calls.
                If None, uses the default fallback chain.
        """
        self.provider = provider
        self._router = LLMRouter()
        self._use_crossencoder = os.environ.get("ENABLE_CROSSENCODER") == "1"

    def load_test_cases(self, path: str | None = None) -> list[TestCase]:
        """Load test cases from JSON file.

        Args:
            path: Path to test cases JSON file. Defaults to eval/test_cases.json
                relative to the project root.

        Returns:
            List of TestCase objects parsed from the JSON file.

        Raises:
            FileNotFoundError: If the test cases file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        file_path = Path(path) if path else Path(DEFAULT_TEST_CASES_PATH)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases: list[TestCase] = []
        for item in data:
            test_cases.append(
                TestCase(
                    post_url=item["post_url"],
                    description=item["description"],
                    expected_themes=item.get("expected_themes", []),
                )
            )

        return test_cases

    async def evaluate_case(self, case: TestCase) -> EvalResult:
        """Run agent on a test case, then score the output.

        Creates an Agent instance, runs the explanation pipeline, extracts
        context for faithfulness scoring, and calls all three judge functions.

        If the agent fails, returns an EvalResult with the error field set.
        If scoring fails, marks error as 'scoring_failed'.

        Args:
            case: The test case to evaluate.

        Returns:
            EvalResult with explanation, scores, reasoning, or error.
        """
        # Step 1: Run the agent pipeline via use case
        extractor = BlueskyExtractor()
        searcher = TavilySearcher()
        ranker = CrossEncoderReranker() if self._use_crossencoder else _ScoreReranker()
        explainer = ExplanationGeneratorAdapter(llm=LLMRouter())
        use_case = ExplainPostUseCase(
            extractor=extractor,
            searcher=searcher,
            ranker=ranker,
            explainer=explainer,
            provider=self.provider,
            verbose=True,
        )
        explanation: ExplanationResult | None = None

        try:
            explanation = await use_case.execute(case.post_url)
        except Exception as e:
            logger.error(
                "Agent failed for %s: %s", case.post_url, str(e)
            )
            return EvalResult(
                test_case=case,
                explanation=None,
                scores=None,
                reasoning={},
                error=f"agent_failed: {type(e).__name__}: {str(e)}",
            )

        # Step 2: Score the output
        try:
            # Build explanation text from bullets
            explanation_text = "\n".join(
                f"• {bullet}" for bullet in explanation.bullets
            )

            # Get context text from use case's last ranked contexts
            context_text = ""
            if use_case.last_ranked_contexts:
                context_text = "\n\n".join(
                    f"[{ctx.title}] ({ctx.url})\n{ctx.content}"
                    for ctx in use_case.last_ranked_contexts
                )

            # Get post text from use case's last post content
            post_text = ""
            if use_case.last_post_content:
                post_text = use_case.last_post_content.text

            # Score all three dimensions
            faithfulness_score, faithfulness_reasoning = await score_faithfulness(
                explanation=explanation_text,
                context=context_text,
                router=self._router,
            )

            relevance_score, relevance_reasoning = await score_relevance(
                explanation=explanation_text,
                post_text=post_text,
                router=self._router,
            )

            helpfulness_score, helpfulness_reasoning = await score_helpfulness(
                explanation=explanation_text,
                post_text=post_text,
                router=self._router,
            )

            scores = EvalScore(
                faithfulness=faithfulness_score,
                answer_relevance=relevance_score,
                context_helpfulness=helpfulness_score,
            )

            reasoning = {
                "faithfulness": faithfulness_reasoning,
                "answer_relevance": relevance_reasoning,
                "context_helpfulness": helpfulness_reasoning,
            }

            return EvalResult(
                test_case=case,
                explanation=explanation,
                scores=scores,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(
                "Scoring failed for %s: %s", case.post_url, str(e)
            )
            return EvalResult(
                test_case=case,
                explanation=explanation,
                scores=None,
                reasoning={},
                error=f"scoring_failed: {type(e).__name__}: {str(e)}",
            )

    async def evaluate_all(
        self, cases: list[TestCase] | None = None
    ) -> list[EvalResult]:
        """Run all test cases and return results.

        Args:
            cases: Optional list of test cases. If None, loads from the
                default test cases file.

        Returns:
            List of EvalResult for each test case.
        """
        if cases is None:
            cases = self.load_test_cases()

        results: list[EvalResult] = []
        for i, case in enumerate(cases, 1):
            logger.info(
                "Evaluating case %d/%d: %s", i, len(cases), case.description
            )
            result = await self.evaluate_case(case)
            results.append(result)

        return results

    def generate_report(self, results: list[EvalResult]) -> dict:
        """Generate aggregate report with per-case scores and overall averages.

        Property 13: averages must equal arithmetic mean of individual scores
        per dimension. Cases with errors are excluded from average calculation.

        Args:
            results: List of EvalResult from evaluate_all.

        Returns:
            Dict suitable for JSON serialization with per-case details
            and aggregate statistics.
        """
        per_case: list[dict] = []
        # Collect scores from successful cases for averaging
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        helpfulness_scores: list[float] = []

        for result in results:
            case_entry: dict = {
                "description": result.test_case.description,
                "post_url": result.test_case.post_url,
            }

            if result.error:
                case_entry["status"] = "error"
                case_entry["error"] = result.error
            elif result.scores:
                case_entry["status"] = "success"
                case_entry["scores"] = {
                    "faithfulness": result.scores.faithfulness,
                    "answer_relevance": result.scores.answer_relevance,
                    "context_helpfulness": result.scores.context_helpfulness,
                }
                # Accumulate for averages (Property 13)
                faithfulness_scores.append(result.scores.faithfulness)
                relevance_scores.append(result.scores.answer_relevance)
                helpfulness_scores.append(result.scores.context_helpfulness)
            else:
                case_entry["status"] = "no_scores"

            per_case.append(case_entry)

        # Compute averages (Property 13: arithmetic mean of individual scores)
        num_scored = len(faithfulness_scores)
        averages: dict = {}
        if num_scored > 0:
            averages = {
                "faithfulness": sum(faithfulness_scores) / num_scored,
                "answer_relevance": sum(relevance_scores) / num_scored,
                "context_helpfulness": sum(helpfulness_scores) / num_scored,
            }
        else:
            averages = {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_helpfulness": 0.0,
            }

        report: dict = {
            "summary": {
                "total_cases": len(results),
                "successful": num_scored,
                "errors": len(results) - num_scored,
                "averages": averages,
            },
            "per_case": per_case,
        }

        return report
