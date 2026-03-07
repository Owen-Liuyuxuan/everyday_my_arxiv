"""
Script to scrape the latest 100 papers from arXiv CV category for testing.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.arxiv.client import ArxivClient


def scrape_test_papers(
    category: str = "cs.CV",
    max_papers: int = 100,
    output_dir: str = "data/test_papers"
) -> List[Dict]:
    """
    Scrape the latest papers from arXiv for testing.

    Args:
        category: ArXiv category to scrape (default: cs.CV)
        max_papers: Maximum number of papers to fetch
        output_dir: Directory to save papers

    Returns:
        List of paper dictionaries
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a custom config for fetching papers
    # We need to bypass the date filter to get the latest papers
    print(f"Scraping latest {max_papers} papers from arXiv category: {category}")

    # Use direct API call to get papers without date restrictions
    base_url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': f'cat:{category}',
        'start': 0,
        'max_results': max_papers,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    url = base_url + '?' + urllib.parse.urlencode(params)
    print(f"Query URL: {url}\n")

    # Fetch data from API
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching data from arXiv API: {e}")
        return []

    # Parse XML response
    papers = parse_arxiv_response(data)

    # Sort by published date (descending)
    papers.sort(key=lambda x: x.get('published_datetime', datetime.min), reverse=True)
    papers = papers[:max_papers]

    print(f"Successfully scraped {len(papers)} papers")

    # Save papers to JSON file
    output_file = os.path.join(output_dir, "papers.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert datetime objects to strings for JSON serialization
        papers_serializable = []
        for paper in papers:
            paper_copy = paper.copy()
            if 'published_datetime' in paper_copy:
                paper_copy['published_datetime'] = paper_copy['published_datetime'].isoformat()
            if 'updated_datetime' in paper_copy:
                paper_copy['updated_datetime'] = paper_copy['updated_datetime'].isoformat()
            papers_serializable.append(paper_copy)

        json.dump(papers_serializable, f, indent=2, ensure_ascii=False)

    print(f"Papers saved to: {output_file}")

    return papers


def parse_arxiv_response(xml_data: str) -> List[Dict]:
    """Parse the XML response from arXiv API."""
    root = ET.fromstring(xml_data)

    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    entries = root.findall('atom:entry', namespaces)
    papers = []

    for entry in entries:
        try:
            entry_id = entry.find('atom:id', namespaces).text
            title = entry.find('atom:title', namespaces).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', namespaces).text.strip().replace('\n', ' ')

            published_str = entry.find('atom:published', namespaces).text
            updated_str = entry.find('atom:updated', namespaces).text

            published_datetime = datetime.fromisoformat(published_str.replace('Z', '+00:00')).replace(tzinfo=None)
            updated_datetime = datetime.fromisoformat(updated_str.replace('Z', '+00:00')).replace(tzinfo=None)

            authors = []
            for author in entry.findall('atom:author', namespaces):
                author_name = author.find('atom:name', namespaces)
                if author_name is not None:
                    authors.append(author_name.text)

            categories = []
            primary_category = None
            for i, category in enumerate(entry.findall('atom:category', namespaces)):
                cat_term = category.get('term')
                if cat_term:
                    categories.append(cat_term)
                    if i == 0:
                        primary_category = cat_term

            pdf_url = None
            abs_url = None
            for link in entry.findall('atom:link', namespaces):
                link_type = link.get('type')
                link_href = link.get('href')
                if link_type == 'application/pdf':
                    pdf_url = link_href.replace('http://', 'https://')
                elif link.get('rel') == 'alternate':
                    abs_url = link_href.replace('http://', 'https://')

            arxiv_id = entry_id.split('/abs/')[-1] if '/abs/' in entry_id else entry_id.split('/')[-1]

            paper = {
                'id': arxiv_id,
                'title': title,
                'authors': authors,
                'abstract': summary,
                'pdf_url': pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                'abs_url': abs_url or f"https://arxiv.org/abs/{arxiv_id}",
                'published_date': published_datetime.strftime('%Y-%m-%d'),
                'updated_date': updated_datetime.strftime('%Y-%m-%d'),
                'published_datetime': published_datetime,
                'updated_datetime': updated_datetime,
                'categories': categories,
                'primary_category': primary_category,
            }

            papers.append(paper)

        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue

    return papers


if __name__ == "__main__":
    papers = scrape_test_papers(category="cs.CV", max_papers=100)
    print(f"\nDone! Scraped {len(papers)} papers.")
