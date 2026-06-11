"""CLI entry point for running the evaluation harness."""

import argparse
import asyncio
import json
import sys

from eval.harness import EvalHarness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-explain",
        description="Run evaluation harness for the Contextual Post Explainer.",
    )
    parser.add_argument(
        "-p", "--provider",
        default=None,
        help="LLM provider/model to use for both agent and judge.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to write JSON report file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Path to test cases JSON file. Defaults to eval/test_cases.json.",
    )
    return parser


def _print_results_table(report: dict) -> None:
    """Print a formatted results table to stdout."""
    summary = report["summary"]
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total cases: {summary['total_cases']}")
    print(f"Successful:  {summary['successful']}")
    print(f"Errors:      {summary['errors']}")
    print()

    avgs = summary["averages"]
    print("Aggregate Scores:")
    print(f"  Faithfulness:         {avgs['faithfulness']:.3f} (0-1)")
    print(f"  Answer Relevance:     {avgs['answer_relevance']:.2f} (1-5)")
    print(f"  Context Helpfulness:  {avgs['context_helpfulness']:.2f} (1-5)")
    print()

    print("-" * 60)
    print(f"{'#':<3} {'Status':<10} {'Faith.':<7} {'Relev.':<7} {'Help.':<7} Description")
    print("-" * 60)

    for i, case in enumerate(report["per_case"], 1):
        status = case["status"]
        desc = case["description"][:35]
        if status == "success":
            scores = case["scores"]
            print(
                f"{i:<3} {'OK':<10} "
                f"{scores['faithfulness']:<7.3f} "
                f"{scores['answer_relevance']:<7.2f} "
                f"{scores['context_helpfulness']:<7.2f} "
                f"{desc}"
            )
        else:
            error_msg = case.get("error", "unknown")[:20]
            print(f"{i:<3} {'ERR':<10} {'—':<7} {'—':<7} {'—':<7} {desc} [{error_msg}]")

    print("-" * 60)


def main() -> None:
    """Entry point: parse args, run evaluation, output results."""
    parser = _build_parser()
    args = parser.parse_args()

    harness = EvalHarness(provider=args.provider)

    # Load test cases
    cases = harness.load_test_cases(path=args.cases)

    # Run evaluation
    results = asyncio.run(harness.evaluate_all(cases=cases))

    # Generate report
    report = harness.generate_report(results)

    # Output results
    _print_results_table(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
