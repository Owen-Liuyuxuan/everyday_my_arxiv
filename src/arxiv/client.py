"""
Arxiv API client for fetching papers based on categories and date filters.
"""
import datetime
import json
import time
from typing import Dict, List, Optional, Tuple

import arxiv
import httpx
from scholarly import scholarly

class ArxivClient:
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the Arxiv client with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)['arxiv']
        
        self.categories = self.config['categories']
        self.max_results = self.config['max_results']
        # Get the base recent_days from config
        self.base_recent_days = self.config['recent_days']
        self.adaptive_recent_days = self.config["adaptive_recent_days"]
        self.citation_lookback_days = self.config['citation_lookback_days']
    
    @property
    def recent_days(self) -> int:
        """
        Get the adaptive recent_days value based on the day of the week.
        If today is Monday in UTC, use 3 days, otherwise use 2 days.
        
        Returns:
            Number of days to look back for recent papers
        """
        if self.adaptive_recent_days:
            today = datetime.datetime.now(datetime.timezone.utc)
            # Monday is 0 in Python's weekday() function
            if today.weekday() == 0:  # If today is Monday
                return 3
            else:
                return 2
        else:
            return self.base_recent_days
    
    def get_recent_papers(self) -> List[Dict]:
        """
        Fetch recent papers from Arxiv based on the configured categories.
        
        Returns:
            List of paper objects with metadata
        """
        # Calculate date range for recent papers
        today = datetime.datetime.now()
        # Use the adaptive recent_days property
        date_filter = today - datetime.timedelta(days=self.recent_days)
        
        print(f"Using adaptive recent_days: {self.recent_days}")
        
        # Construct the query for Arxiv API
        query = " OR ".join([f"cat:{category}" for category in self.categories])
        
        # Fetch papers from Arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in client.results(search):
            # Convert to datetime for comparison
            published_date = result.published.replace(tzinfo=None)

            # print(f"title: {result.title}, published_date: {published_date}")
            
            # Only include papers within the date range
            if published_date >= date_filter:
                paper = {
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'pdf_url': result.pdf_url.replace("http", "https"),
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'updated_date': result.updated.strftime('%Y-%m-%d'),
                    'categories': result.categories,
                    'comment': getattr(result, 'comment', ''),
                    'journal_ref': getattr(result, 'journal_ref', ''),
                    'doi': getattr(result, 'doi', ''),
                    'primary_category': result.primary_category
                }
                papers.append(paper)
        
        print(f"Found {len(papers)} recent papers in the specified categories")
        return papers
    
    def get_citation_data(self, papers: List[Dict], max_papers: int = 20) -> List[Dict]:
        """
        Fetch citation data for papers using Google Scholar.
        
        Args:
            papers: List of paper objects
            max_papers: Maximum number of papers to check (to avoid rate limiting)
            
        Returns:
            Updated list of papers with citation data
        """
        # Limit the number of papers to check to avoid rate limiting
        papers_to_check = papers[:max_papers]
        
        for i, paper in enumerate(papers_to_check):
            print(f"Checking citations for paper {i+1}/{len(papers_to_check)}: {paper['title']}")
            
            try:
                # Search for the paper on Google Scholar
                query = f"{paper['title']} {paper['authors'][0]}"
                search_query = scholarly.search_pubs(query)
                result = next(search_query, None)
                
                if result:
                    # Get citation data
                    paper['citation_count'] = result.get('num_citations', 0)
                    paper['citation_url'] = result.get('citedby_url', '')
                else:
                    paper['citation_count'] = 0
                    paper['citation_url'] = ''
                    
                # Add a small delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"Error fetching citation data: {e}")
                paper['citation_count'] = 0
                paper['citation_url'] = ''
        
        return papers
    
    def get_pdf_content(self, pdf_url: str) -> bytes:
        """
        Download PDF content from a URL.
        
        Args:
            pdf_url: URL to the PDF file
            
        Returns:
            PDF content as bytes or empty bytes if the file is too large (>20MB) or download fails
        """
        try:
            # First make a HEAD request to check the file size
            head_response = httpx.head(pdf_url, timeout=10.0)
            head_response.raise_for_status()
            
            # Check if Content-Length header exists
            content_length = head_response.headers.get('Content-Length')
            if content_length:
                # Convert to integer (bytes)
                file_size = int(content_length)
                # 20MB = 20 * 1024 * 1024 bytes
                if file_size > 20 * 1024 * 1024:
                    print(f"PDF from {pdf_url} is too large ({file_size / (1024 * 1024):.2f} MB), skipping download")
                    return b''
            
            # If size is acceptable or unknown, proceed with download
            response = httpx.get(pdf_url, timeout=30.0)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading PDF from {pdf_url}: {e}")
            return b''
    
    def get_most_cited_papers(self, days: Optional[int] = None) -> List[Dict]:
        """
        Get the most cited papers from the past specified days.
        
        Args:
            days: Number of days to look back (defaults to config value)
            
        Returns:
            List of paper objects sorted by citation count
        """
        if days is None:
            days = self.citation_lookback_days
            
        # Calculate date range
        today = datetime.datetime.now()
        date_filter = today - datetime.timedelta(days=days)
        
        # Construct the query for Arxiv API
        query = " OR ".join([f"cat:{category}" for category in self.categories])
        
        # Fetch papers from Arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=200,  # Get more papers to find the most cited ones
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in client.results(search):
            # Convert to datetime for comparison
            published_date = result.published.replace(tzinfo=None)
            
            # Only include papers within the date range
            if published_date >= date_filter:
                paper = {
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'pdf_url': result.pdf_url.replace("http", "https"),
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'categories': result.categories,
                    'primary_category': result.primary_category
                }
                papers.append(paper)
        
        # Get citation data for these papers
        papers_with_citations = self.get_citation_data(papers)
        
        # Sort by citation count
        sorted_papers = sorted(
            papers_with_citations, 
            key=lambda x: x.get('citation_count', 0), 
            reverse=True
        )
        
        return sorted_papers