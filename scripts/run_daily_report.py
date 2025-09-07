#!/usr/bin/env python3
"""
Main script to run the daily Arxiv paper report with enhanced paper selection using
Gemini AI to determine relevance and significance.
"""
import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arxiv.client import ArxivClient
from src.arxiv.parser import ArxivParser
from src.llm.gemini import GeminiClient
from src.output.markdown import MarkdownReportGenerator
from src.output.email import EmailNotifier
from src.utils.citation import CitationAnalyzer
from src.utils.filters import PaperFilter
from src.utils.ranking import PaperRanker

def main():
    """Run the daily Arxiv paper report with enhanced AI-based paper selection."""
    parser = argparse.ArgumentParser(description="Generate daily Arxiv paper report")
    parser.add_argument("--config", default="config/config.json", help="Path to config file")
    parser.add_argument("--keywords", default="config/keywords.json", help="Path to keywords file")
    parser.add_argument("--date", help="Report date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--no-email", action="store_true", help="Disable email notification")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip AI-based paper scoring")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load keywords
    with open(args.keywords, 'r') as f:
        keywords_config = json.load(f)
        keywords = keywords_config.get('primary_keywords', []) + keywords_config.get('secondary_keywords', [])
        exclude_keywords = keywords_config.get('exclude_keywords', [])
    
    # Set date
    if args.date:
        report_date = args.date
    else:
        report_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    print(f"Generating Arxiv paper report for {report_date}")
    
    # Initialize components
    arxiv_client = ArxivClient(config_path=args.config)
    arxiv_parser = ArxivParser(keywords_path=args.keywords)
    gemini_client = GeminiClient(config_path=args.config)
    markdown_generator = MarkdownReportGenerator(config_path=args.config)
    email_notifier = EmailNotifier(config_path=args.config)
    paper_filter = PaperFilter()
    paper_ranker = PaperRanker(min_combined_score=config.get('ranking', {}).get('min_combined_score', 4))
    
    # 1. Fetch recent papers
    print("Fetching recent papers from Arxiv...")
    recent_papers = arxiv_client.get_recent_papers()
    
    # 2. Filter papers by date
    recent_papers = paper_filter.filter_by_date(recent_papers, days=arxiv_client.recent_days)
    print(f"Found {len(recent_papers)} papers published in the last {arxiv_client.recent_days} days")
    
    # 3. Filter papers by category
    recent_papers = paper_filter.filter_by_category(recent_papers, categories=config['arxiv']['categories'])
    print(f"Found {len(recent_papers)} papers in the specified categories")
    
    # 4. Initial filtering using keyword matching
    # Score papers using Gemini for relevance and significance
    print("Scoring papers for relevance and significance using Gemini...")
    scored_papers = gemini_client.batch_score_papers(
        recent_papers, 
        keywords=keywords,
        negative_keywords=exclude_keywords
    )
    
    # 7. Select top papers based on combined score
    print("Selecting top papers based on relevance and significance scores...")
    max_papers = config['report']['max_papers']
    selected_papers = paper_ranker.select_top_papers(scored_papers, limit=max_papers)
    print(f"Selected {len(selected_papers)} papers for the report")
    
    # 7. Analyze papers using Gemini
    print("Analyzing papers with Gemini...")
    for i, paper in enumerate(selected_papers):
        print(f"Analyzing paper {i+1}/{len(selected_papers)}: {paper['title']}")
        
        # Enrich paper data
        paper = arxiv_parser.enrich_paper_data(paper)
        
        # Try to get the full PDF for better analysis
        pdf_data = arxiv_client.get_pdf_content(paper['pdf_url'])
        
        if pdf_data:
            # Analyze using the full PDF
            paper['analysis'] = gemini_client.analyze_paper_from_pdf(pdf_data, paper)
        else:
            # Fall back to abstract-based analysis
            paper['analysis'] = gemini_client.analyze_paper_from_abstract(paper)
    
    # 8. Generate report summary
    print("Generating report summary...")
    report_summary = gemini_client.generate_report_summary(selected_papers, report_type="daily")
    
    # 9. Generate Markdown report
    print("Generating Markdown report...")
    markdown_report = markdown_generator.generate_daily_report(
        papers=selected_papers,
        report_summary=report_summary,
        date=report_date
    )
    
    # 10. Save report
    report_filename = f"arxiv_cv_report_{report_date}.md"
    markdown_path = markdown_generator.save_report(markdown_report, filename=report_filename)
    
    # 11. Convert to HTML if needed
    html_path = None
    if 'html' in config['report']['output_format']:
        print("Converting report to HTML...")
        html_path = markdown_generator.convert_to_html(markdown_path)
    
    # 12. Send email notification if enabled
    if not args.no_email and config['email']['enabled']:
        print("Sending email notification...")
        email_notifier.send_report_notification(
            date=report_date,
            paper_count=len(selected_papers),
            report_summary=report_summary,
            markdown_report_path=markdown_path,
            html_report_path=html_path
        )
    
    print(f"Report generation completed. Report saved to {markdown_path}")
    if html_path:
        print(f"HTML report saved to {html_path}")

if __name__ == "__main__":
    main()