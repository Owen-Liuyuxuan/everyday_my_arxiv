"""
Script to record the current scoring for test papers.
This script loads papers from data/test_papers/papers.json and runs scoring.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv

# Add src to path for imports (must be before src.* imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm.factory import create_scoring_client

# Load environment variables from .env file
load_dotenv()




def load_papers(papers_file: str = "data/test_papers/papers.json") -> List[Dict]:
    """Load papers from JSON file."""
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # Convert datetime strings back to datetime objects
    for paper in papers:
        if 'published_datetime' in paper and isinstance(paper['published_datetime'], str):
            paper['published_datetime'] = datetime.fromisoformat(paper['published_datetime'])
        if 'updated_datetime' in paper and isinstance(paper['updated_datetime'], str):
            paper['updated_datetime'] = datetime.fromisoformat(paper['updated_datetime'])

    return papers


def load_keywords(keywords_file: str = "config/keywords.json") -> Dict:
    """Load keywords from config."""
    with open(keywords_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def record_scoring(
    papers_file: str = "data/test_papers/papers.json",
    keywords_file: str = "config/keywords.json",
    output_file: str = "data/test_papers/scoring_results.json",
    config_path: str = "config/config.json"
) -> List[Dict]:
    """
    Record current scoring for test papers.

    Args:
        papers_file: Path to papers JSON file
        keywords_file: Path to keywords JSON file
        output_file: Path to save scoring results
        config_path: Path to config file for LLM provider selection

    Returns:
        List of papers with scoring results
    """
    print("Loading papers...")
    papers = load_papers(papers_file)
    print(f"Loaded {len(papers)} papers")

    print("Loading keywords...")
    keywords_config = load_keywords(keywords_file)
    primary_keywords = keywords_config.get('primary_keywords', [])
    exclude_keywords = keywords_config.get('exclude_keywords', [])
    print(f"Primary keywords: {len(primary_keywords)}")
    print(f"Exclude keywords: {len(exclude_keywords)}")

    print("\nInitializing scoring client from config...")
    client = create_scoring_client(config_path=config_path)
    print(f"Using scoring provider: {client.__class__.__name__}")

    print(f"\nScoring {len(papers)} papers...")
    scored_papers = client.batch_score_papers(papers, primary_keywords, exclude_keywords)

    # Save results
    print(f"\nSaving scoring results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Prepare serializable results
    results = []
    for paper in scored_papers:
        result = {
            'id': paper.get('id'),
            'title': paper.get('title'),
            'authors': paper.get('authors'),
            'published_date': paper.get('published_date'),
            'relevance_score': paper.get('relevance_score'),
            'significance_score': paper.get('significance_score'),
            'combined_score': paper.get('combined_score'),
        }
        results.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    scores = [p.get('combined_score', 0) for p in scored_papers]
    print(f"\nScoring Summary:")
    print(f"  Total papers: {len(scored_papers)}")
    print(f"  Average combined score: {sum(scores)/len(scores):.2f}")
    print(f"  Min score: {min(scores)}")
    print(f"  Max score: {max(scores)}")
    print(f"  Papers with score >= 4: {sum(1 for s in scores if s >= 4)}")

    print(f"\nResults saved to: {output_file}")

    return scored_papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record scoring for test papers")
    parser.add_argument("--config", default="config/config.json",
                        help="Path to config file for LLM provider")
    parser.add_argument("--papers", default="data/test_papers/papers.json",
                        help="Path to papers JSON file")
    parser.add_argument("--keywords", default="config/keywords.json",
                        help="Path to keywords JSON file")
    parser.add_argument("--output", default="data/test_papers/scoring_results.json",
                        help="Path to output file")
    args = parser.parse_args()

    results = record_scoring(
        papers_file=args.papers,
        keywords_file=args.keywords,
        output_file=args.output,
        config_path=args.config
    )
    print(f"\nDone! Scored {len(results)} papers.")
