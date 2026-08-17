#!/usr/bin/env python3
"""Run one local paper through the daily report's real OpenAI call sequence."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.llm.factory import create_pdf_client, create_scoring_client


DEFAULT_CONFIG = ROOT / "config" / "config.json"
DEFAULT_PDF = ROOT / "examples" / "openai_pdf_sample.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make the same scoring, PDF-analysis, and report-summary API calls as "
            "the daily workflow, using one local fixture. This makes three billable calls."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser.parse_args()


def sample_paper() -> dict:
    return {
        "title": "OpenAI PDF Reading Smoke Test",
        "authors": ["Arxiv Analyzer Test"],
        "formatted_authors": "Arxiv Analyzer Test",
        "abstract": (
            "A local integration fixture for validating paper scoring, PDF reading, "
            "and daily report summarization."
        ),
        "categories": ["cs.RO", "cs.CV"],
        "published_date": "2026-08-17",
        "venue": "Local smoke test",
        "code_url": "N/A",
    }


def read_pdf(path: Path) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PDF not found: {resolved}")
    data = resolved.read_bytes()
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError(f"File does not have a PDF header: {resolved}")
    return data


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    config_path = str(args.config.expanduser().resolve())
    scoring_client = create_scoring_client(config_path)
    pdf_client = create_pdf_client(config_path)
    paper = sample_paper()

    print(f"[1/3] Scoring with {scoring_client.model_name}...")
    scored_paper = scoring_client.batch_score_papers(
        [paper],
        keywords=["robotics", "computer vision", "autonomous driving"],
    )[0]
    print(
        "Scores: "
        f"relevance={scored_paper['relevance_score']}, "
        f"significance={scored_paper['significance_score']}, "
        f"combined={scored_paper['combined_score']}"
    )

    print(f"[2/3] Analyzing local PDF with {pdf_client.model_name}...")
    scored_paper["analysis"] = pdf_client.analyze_paper_from_pdf(
        read_pdf(args.pdf), scored_paper
    )
    print(scored_paper["analysis"])

    print(f"[3/3] Generating report summary with {scoring_client.model_name}...")
    report_summary = scoring_client.generate_report_summary([scored_paper])
    print(report_summary)
    print("Daily OpenAI flow smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
