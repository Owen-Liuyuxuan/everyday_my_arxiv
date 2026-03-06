#!/usr/bin/env python3
"""
Main script to run the daily Arxiv paper report with enhanced paper selection using
LLM (Gemini or Ark) to determine relevance and significance.
Modified to support granular stage execution for OpenClaw Agent migration.
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
from src.llm.factory import create_llm_client
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
    parser.add_argument("--provider", choices=["gemini", "ark"], default=None,
                        help="LLM provider to use (auto-detected from config path if not specified)")
    parser.add_argument("--stage", choices=["fetch", "score", "analyze", "report", "all"], default="all",
                        help="Granular stage to execute (for agentic migration)")
    parser.add_argument("--input-file", help="Path to input JSON file for a specific stage")
    parser.add_argument("--output-file", help="Path to output JSON file for a specific stage")
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
    
    # Initialize components
    arxiv_client = ArxivClient(config_path=args.config)
    arxiv_parser = ArxivParser(keywords_path=args.keywords)
    markdown_generator = MarkdownReportGenerator(config_path=args.config)
    email_notifier = EmailNotifier(config_path=args.config)
    paper_filter = PaperFilter()
    paper_ranker = PaperRanker(min_combined_score=config.get('ranking', {}).get('min_combined_score', 4))

    llm_client = None
    if args.stage in ["score", "analyze", "report", "all"]:
        llm_client, provider_name = create_llm_client(config_path=args.config, provider=args.provider)
        print(f"Using LLM provider: {provider_name}")

    papers = []

    # --- STAGE: FETCH ---
    if args.stage == "fetch" or args.stage == "all":
        print(f"Generating Arxiv paper report for {report_date}")
        print("Fetching recent papers from Arxiv...")
        recent_papers = arxiv_client.get_recent_papers()
        recent_papers = paper_filter.filter_by_date(recent_papers, days=arxiv_client.recent_days)
        print(f"Found {len(recent_papers)} papers published in the last {arxiv_client.recent_days} days")
        recent_papers = paper_filter.filter_by_category(recent_papers, categories=config['arxiv']['categories'])
        print(f"Found {len(recent_papers)} papers in the specified categories")
        papers = recent_papers
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(papers, f, indent=2)
            print(f"Fetch stage completed. Output saved to {args.output_file}")
            if args.stage == "fetch": return

    # --- STAGE: SCORE ---
    if args.stage == "score" or args.stage == "all":
        # Only load input file if we are running the 'score' stage standalone.
        # If running 'all', use the results from the previous 'fetch' stage in memory.
        if args.input_file and args.stage == "score":
            with open(args.input_file, 'r') as f:
                papers = json.load(f)
        
        print("Scoring papers for relevance and significance using LLM...")
        papers = llm_client.batch_score_papers(
            papers, 
            keywords=keywords,
            negative_keywords=exclude_keywords
        )
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(papers, f, indent=2)
            print(f"Score stage completed. Output saved to {args.output_file}")
            if args.stage == "score": return

    # --- STAGE: ANALYZE ---
    if args.stage == "analyze" or args.stage == "all":
        # Only load input file if we are running the 'analyze' stage standalone.
        if args.input_file and args.stage == "analyze":
            with open(args.input_file, 'r') as f:
                papers = json.load(f)
        
        print("Selecting top papers based on relevance and significance scores...")
        max_papers = config['report']['max_papers']
        selected_papers = paper_ranker.select_top_papers(papers, limit=max_papers)
        print(f"Selected {len(selected_papers)} papers for analysis")
        
        print("Analyzing papers with LLM...")
        for i, paper in enumerate(selected_papers):
            print(f"Analyzing paper {i+1}/{len(selected_papers)}: {paper['title']}")
            paper = arxiv_parser.enrich_paper_data(paper)
            pdf_data = arxiv_client.get_pdf_content(paper['pdf_url'])
            
            if pdf_data:
                paper['analysis'] = llm_client.analyze_paper_from_pdf(pdf_data, paper)
            else:
                print(f"Falling back to abstract-based analysis for {paper['title']}")
                paper['analysis'] = llm_client.analyze_paper_from_abstract(paper)
        
        papers = selected_papers
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(papers, f, indent=2)
            print(f"Analyze stage completed. Output saved to {args.output_file}")
            if args.stage == "analyze": return

    # --- STAGE: REPORT ---
    if args.stage == "report" or args.stage == "all":
        # Only load input file if we are running the 'report' stage standalone.
        if args.input_file and args.stage == "report":
            with open(args.input_file, 'r') as f:
                papers = json.load(f)
        
        print("Generating report summary...")
        report_summary = llm_client.generate_report_summary(papers, report_type="daily")
        
        print("Generating Markdown report...")
        markdown_report = markdown_generator.generate_daily_report(
            papers=papers,
            report_summary=report_summary,
            date=report_date
        )
        
        report_filename = f"arxiv_cv_report_{report_date}.md"
        markdown_path = markdown_generator.save_report(markdown_report, filename=report_filename)
        
        html_path = None
        if 'html' in config['report']['output_format']:
            print("Converting report to HTML...")
            html_path = markdown_generator.convert_to_html(markdown_path)
        
        if not args.no_email and config['email']['enabled']:
            print("Sending email notification...")
            email_notifier.send_report_notification(
                date=report_date,
                paper_count=len(papers),
                report_summary=report_summary,
                markdown_report_path=markdown_path,
                html_report_path=html_path
            )
        
        print(f"Report generation completed. Report saved to {markdown_path}")

if __name__ == "__main__":
    main()
