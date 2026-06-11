"""LLM-as-judge scoring functions for the RAG Triad evaluation.

Implements three scoring dimensions:
- Faithfulness: claim decomposition + entailment verification (0.0-1.0)
- Answer Relevance: how directly the explanation addresses the post (1.0-5.0)
- Context Helpfulness: multi-dimensional quality grading (1.0-5.0)

Each function uses structured rubrics and returns (score, reasoning) tuples
for transparency and debugging.
"""

import re

from src.adapters.litellm_router import LiteLLMRouter as LLMRouter


FAITHFULNESS_DECOMPOSE_PROMPT = """You are a claim verification expert.

Decompose the following explanation into atomic claims. Each claim should be a single, verifiable factual statement.

Explanation:
{explanation}

List each atomic claim on its own numbered line. Output ONLY the numbered list, nothing else.
Example format:
1. Claim one here
2. Claim two here
"""

FAITHFULNESS_VERIFY_PROMPT = """You are a claim verification expert.

For each claim below, determine if it is supported by the provided context.
Answer YES if the claim is directly supported or can be reasonably inferred from the context.
Answer NO if the claim is not supported or contradicts the context.

Context:
{context}

Claims to verify:
{claims}

For each claim, respond with ONLY the claim number and YES or NO.
Example format:
1. YES
2. NO
3. YES
"""

RELEVANCE_PROMPT = """You are an answer relevance evaluator.

Rate on a scale of 1-5 how directly this explanation addresses the core content and intent of the original post.

Rubric:
1 = Completely irrelevant - the explanation has nothing to do with the post
2 = Mostly irrelevant - touches on tangential topics but misses the core point
3 = Partially relevant - addresses some aspects but misses key content
4 = Mostly relevant - addresses the main points with minor gaps
5 = Perfectly relevant - directly and completely addresses the post's core content

Original Post:
{post_text}

Explanation:
{explanation}

Provide your reasoning first, then on the final line output ONLY the numeric score.
Example final line: 4
"""

HELPFULNESS_PROMPT = """You are an explanation quality evaluator.

Evaluate the following explanation on 5 sub-dimensions, each on a 1-5 Likert scale.

Sub-dimensions:
1. Credibility: Are claims properly sourced or attributable? (1=unsourced speculation, 5=well-sourced claims)
2. Clarity: Is the explanation easy to understand? (1=confusing/jargon-heavy, 5=crystal clear)
3. Relevance: Does it address what the reader needs to know? (1=misses the point, 5=exactly what's needed)
4. Veracity: Are statements factually accurate? (1=contains falsehoods, 5=fully accurate)
5. Neutrality: Is the tone balanced and objective? (1=heavily biased, 5=perfectly balanced)

Original Post:
{post_text}

Explanation:
{explanation}

For each dimension, provide brief reasoning and a score. Then output a summary line with all 5 scores.
Use this exact format for the summary line:
SCORES: credibility=X, clarity=X, relevance=X, veracity=X, neutrality=X
"""


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to the given range."""
    return max(min_val, min(max_val, value))


def _extract_number(text: str, min_val: float, max_val: float) -> float:
    """Extract a numeric score from text, searching from the end.

    Looks for standalone numbers (integers or decimals) in the text,
    preferring the last one found (which is typically the final score).
    """
    # Look for numbers at the end of the text first
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    if not numbers:
        # Fallback: return midpoint of range
        return (min_val + max_val) / 2

    # Take the last number found (usually the final score line)
    for num_str in reversed(numbers):
        value = float(num_str)
        if min_val <= value <= max_val:
            return value

    # If no number in range, clamp the last number found
    return _clamp(float(numbers[-1]), min_val, max_val)


def _parse_yes_no_counts(text: str) -> tuple[int, int]:
    """Parse YES/NO responses and return (yes_count, total_count)."""
    lines = text.strip().split("\n")
    yes_count = 0
    total = 0

    for line in lines:
        line_upper = line.strip().upper()
        if "YES" in line_upper or "NO" in line_upper:
            total += 1
            if "YES" in line_upper:
                yes_count += 1

    return yes_count, total


def _parse_helpfulness_scores(text: str) -> list[float]:
    """Parse the SCORES line from helpfulness evaluation.

    Looks for pattern: SCORES: credibility=X, clarity=X, ...
    Falls back to extracting any 5 numbers near the end.
    """
    # Try to find the SCORES: line
    scores_match = re.search(
        r"SCORES:\s*credibility\s*=\s*(\d+(?:\.\d+)?)\s*,\s*"
        r"clarity\s*=\s*(\d+(?:\.\d+)?)\s*,\s*"
        r"relevance\s*=\s*(\d+(?:\.\d+)?)\s*,\s*"
        r"veracity\s*=\s*(\d+(?:\.\d+)?)\s*,\s*"
        r"neutrality\s*=\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if scores_match:
        return [float(scores_match.group(i)) for i in range(1, 6)]

    # Fallback: look for patterns like "X/5" or standalone numbers near dimension names
    dimensions = ["credibility", "clarity", "relevance", "veracity", "neutrality"]
    scores = []
    for dim in dimensions:
        # Search for "dimension: X" or "dimension = X" patterns
        pattern = rf"{dim}\s*[:=]\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scores.append(float(match.group(1)))

    if len(scores) == 5:
        return scores

    # Last fallback: find any 5 numbers between 1-5 near the end
    numbers = re.findall(r"\b([1-5](?:\.\d+)?)\b", text)
    if len(numbers) >= 5:
        return [float(n) for n in numbers[-5:]]

    # Return midpoint defaults if parsing fails
    return [3.0, 3.0, 3.0, 3.0, 3.0]


async def score_faithfulness(
    explanation: str, context: str, router: LLMRouter
) -> tuple[float, str]:
    """Decompose explanation into atomic claims, verify each against context.

    Returns (ratio of supported claims / total claims, reasoning string).
    Score range: 0.0 to 1.0 (Property 12).

    Args:
        explanation: The explanation text to evaluate.
        context: The retrieved context to verify claims against.
        router: LLMRouter instance for LLM calls.

    Returns:
        Tuple of (faithfulness_score, reasoning_text).
    """
    # Step 1: Decompose explanation into atomic claims
    decompose_messages = [
        {"role": "system", "content": "You are a claim verification expert."},
        {
            "role": "user",
            "content": FAITHFULNESS_DECOMPOSE_PROMPT.format(explanation=explanation),
        },
    ]
    decompose_response = await router.completion(decompose_messages)
    claims_text = decompose_response["content"].strip()

    # Check if we got any claims
    claims_lines = [
        line.strip()
        for line in claims_text.split("\n")
        if line.strip() and re.match(r"^\d+[\.\)]\s*", line.strip())
    ]

    if not claims_lines:
        # No claims extracted - treat as fully faithful (vacuous truth)
        return 1.0, "No atomic claims could be extracted from the explanation."

    # Step 2: Verify each claim against context
    numbered_claims = "\n".join(claims_lines)
    verify_messages = [
        {"role": "system", "content": "You are a claim verification expert."},
        {
            "role": "user",
            "content": FAITHFULNESS_VERIFY_PROMPT.format(
                context=context, claims=numbered_claims
            ),
        },
    ]
    verify_response = await router.completion(verify_messages)
    verify_text = verify_response["content"].strip()

    # Parse YES/NO counts
    yes_count, total = _parse_yes_no_counts(verify_text)

    # Use the number of claims from decomposition as total if parsing found fewer
    total = max(total, len(claims_lines))
    if total == 0:
        return 1.0, "No claims to verify."

    # If verification didn't produce enough responses, assume unverified = NO
    score = _clamp(yes_count / total, 0.0, 1.0)

    reasoning = (
        f"Decomposed into {len(claims_lines)} atomic claims. "
        f"Verified {yes_count}/{total} claims are supported by context.\n\n"
        f"Claims:\n{numbered_claims}\n\n"
        f"Verification:\n{verify_text}"
    )

    return score, reasoning


async def score_relevance(
    explanation: str, post_text: str, router: LLMRouter
) -> tuple[float, str]:
    """Measure how directly the explanation addresses the post's core content.

    Returns (score 1-5, reasoning string).
    Score range: 1.0 to 5.0 (Property 12).

    Args:
        explanation: The explanation text to evaluate.
        post_text: The original post text.
        router: LLMRouter instance for LLM calls.

    Returns:
        Tuple of (relevance_score, reasoning_text).
    """
    messages = [
        {"role": "system", "content": "You are an answer relevance evaluator."},
        {
            "role": "user",
            "content": RELEVANCE_PROMPT.format(
                post_text=post_text, explanation=explanation
            ),
        },
    ]
    response = await router.completion(messages)
    response_text = response["content"].strip()

    score = _extract_number(response_text, 1.0, 5.0)
    score = _clamp(score, 1.0, 5.0)

    reasoning = f"Relevance evaluation:\n{response_text}"

    return score, reasoning


async def score_helpfulness(
    explanation: str, post_text: str, router: LLMRouter
) -> tuple[float, str]:
    """Grade on 5 sub-dimensions: credibility, clarity, relevance, veracity, neutrality.

    Each sub-dimension 1-5 Likert scale. Returns (average, reasoning string).
    Score range: 1.0 to 5.0 (Property 12).

    Args:
        explanation: The explanation text to evaluate.
        post_text: The original post text.
        router: LLMRouter instance for LLM calls.

    Returns:
        Tuple of (average_score, reasoning_text).
    """
    messages = [
        {"role": "system", "content": "You are an explanation quality evaluator."},
        {
            "role": "user",
            "content": HELPFULNESS_PROMPT.format(
                post_text=post_text, explanation=explanation
            ),
        },
    ]
    response = await router.completion(messages)
    response_text = response["content"].strip()

    scores = _parse_helpfulness_scores(response_text)

    # Clamp each sub-dimension to [1.0, 5.0]
    clamped_scores = [_clamp(s, 1.0, 5.0) for s in scores]
    average = sum(clamped_scores) / len(clamped_scores)

    # Final clamp to ensure Property 12 invariant
    average = _clamp(average, 1.0, 5.0)

    dimension_names = ["credibility", "clarity", "relevance", "veracity", "neutrality"]
    score_breakdown = ", ".join(
        f"{name}={score:.1f}" for name, score in zip(dimension_names, clamped_scores)
    )

    reasoning = (
        f"Sub-dimension scores: {score_breakdown}\n"
        f"Average: {average:.2f}\n\n"
        f"Full evaluation:\n{response_text}"
    )

    return average, reasoning
