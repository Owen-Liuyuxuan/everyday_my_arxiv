#!/usr/bin/env python3
"""Smoke-test the repository's OpenAI PDF path with a local PDF file."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.llm.factory import create_pdf_client


DEFAULT_PDF = ROOT / "examples" / "openai_pdf_sample.pdf"
DEFAULT_CONFIG = ROOT / "config" / "config_openai_pdf.json"
MAX_PDF_BYTES = 50 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local PDF to OpenAI and print a short grounded reading test."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Local PDF path (default: {DEFAULT_PDF.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"OpenAI configuration (default: {DEFAULT_CONFIG.relative_to(ROOT)})",
    )
    return parser.parse_args()


def read_local_pdf(path: Path) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PDF not found: {resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {resolved}")

    size = resolved.stat().st_size
    if size >= MAX_PDF_BYTES:
        raise ValueError(f"PDF must be smaller than 50 MB: {resolved}")

    data = resolved.read_bytes()
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError(f"File does not have a PDF header: {resolved}")
    return data


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Export it or add it to the ignored .env file, "
            "then rerun this command.",
            file=sys.stderr,
        )
        return 2

    pdf_path = args.pdf.expanduser().resolve()
    pdf_data = read_local_pdf(pdf_path)
    client = create_pdf_client(str(args.config.expanduser().resolve()))

    print(f"Local PDF: {pdf_path} ({len(pdf_data) / 1024:.1f} KiB)")
    print(f"OpenAI model: {client.model_name}")
    print("Uploading PDF and requesting a grounded reading test...")

    result = client.analyze_paper_from_pdf(
        pdf_data,
        paper_metadata={"title": pdf_path.stem},
        prompt_type="pdf_smoke_test",
    )
    print("\nOpenAI response:\n")
    print(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
