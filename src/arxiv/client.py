"""
Arxiv API client for fetching papers based on categories and date filters.
"""
import datetime
import json
import time
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse

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

        self.base_url = 'http://export.arxiv.org/api/query'
    
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
        Fetch papers from arXiv with comprehensive field extraction.
        
        Args:
            categories: List of arXiv categories to search
            days_back: Number of days to look back (optional)
            date_filter: Only include papers published after this date (optional)
            
        Returns:
            List of paper dictionaries with all relevant fields
        """
        # Build the search query
        cat_query = ' OR '.join([f'cat:{cat}' for cat in self.categories])
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=self.recent_days)
        start_str = start_date.strftime('%Y%m%d') + '0000'
        end_str = end_date.strftime('%Y%m%d') + '2359'
        search_query = f'({cat_query}) AND submittedDate:[{start_str} TO {end_str}]'

        
        # Build URL parameters
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': self.max_results + 100,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        url = self.base_url + '?' + urllib.parse.urlencode(params)
        
        print(f"Fetching papers from arXiv API...")
        print(f"Categories: {', '.join(self.categories)}")
        print(f"Date range: Last {self.recent_days} days")
        print(f"Query URL: {url}\n")
        
        # Fetch data from API
        try:
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')
        except Exception as e:
            print(f"Error fetching data from arXiv API: {e}")
            return []
        
        # Parse XML response
        papers = self._parse_arxiv_response(data)
        
        # Sort by published date (descending) - CLIENT-SIDE SORTING
        papers.sort(key=lambda x: x['published_datetime'], reverse=True)
        papers = papers[:self.max_results]
        
        print(f"Successfully fetched and sorted {len(papers)} papers\n")
        
        return papers
    
    def _parse_arxiv_response(
        self, 
        xml_data: str) -> List[Dict]:
        """
        Parse the XML response from arXiv API and extract all relevant fields.
        
        Args:
            xml_data: XML response string from arXiv API
            date_filter: Optional datetime filter for published date
            
        Returns:
            List of paper dictionaries
        """
        root = ET.fromstring(xml_data)
        
        # Define XML namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entries = root.findall('atom:entry', namespaces)
        papers = []
        
        for entry in entries:
            try:
                # Extract basic fields
                entry_id = entry.find('atom:id', namespaces).text
                title = entry.find('atom:title', namespaces).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', namespaces).text.strip().replace('\n', ' ')
                
                # Extract dates
                published_str = entry.find('atom:published', namespaces).text
                updated_str = entry.find('atom:updated', namespaces).text
                
                # Parse dates
                published_datetime = datetime.datetime.fromisoformat(published_str.replace('Z', '+00:00')).replace(tzinfo=None)
                updated_datetime = datetime.datetime.fromisoformat(updated_str.replace('Z', '+00:00')).replace(tzinfo=None)
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    author_name = author.find('atom:name', namespaces)
                    if author_name is not None:
                        authors.append(author_name.text)
                
                # Extract categories
                categories = []
                primary_category = None
                for i, category in enumerate(entry.findall('atom:category', namespaces)):
                    cat_term = category.get('term')
                    if cat_term:
                        categories.append(cat_term)
                        if i == 0:  # First category is primary
                            primary_category = cat_term
                
                # Extract links
                pdf_url = None
                abs_url = None
                for link in entry.findall('atom:link', namespaces):
                    link_type = link.get('type')
                    link_href = link.get('href')
                    if link_type == 'application/pdf':
                        pdf_url = link_href.replace('http://', 'https://')
                    elif link.get('rel') == 'alternate':
                        abs_url = link_href.replace('http://', 'https://')
                
                # Extract optional arXiv-specific fields
                comment_elem = entry.find('arxiv:comment', namespaces)
                comment = comment_elem.text if comment_elem is not None else ''
                
                journal_ref_elem = entry.find('arxiv:journal_ref', namespaces)
                journal_ref = journal_ref_elem.text if journal_ref_elem is not None else ''
                
                doi_elem = entry.find('arxiv:doi', namespaces)
                doi = doi_elem.text if doi_elem is not None else ''
                
                primary_category_elem = entry.find('arxiv:primary_category', namespaces)
                if primary_category_elem is not None:
                    primary_category = primary_category_elem.get('term', primary_category)
                
                # Extract arXiv ID from entry_id
                arxiv_id = entry_id.split('/abs/')[-1] if '/abs/' in entry_id else entry_id.split('/')[-1]
                
                # Build paper dictionary matching the arxiv library format
                paper = {
                    'id': arxiv_id,
                    'title': title,
                    'authors': authors,
                    'abstract': summary,
                    'pdf_url': pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                    'abs_url': abs_url or f"https://arxiv.org/abs/{arxiv_id}",
                    'published_date': published_datetime.strftime('%Y-%m-%d'),
                    'updated_date': updated_datetime.strftime('%Y-%m-%d'),
                    'published_datetime': published_datetime,  # Keep for sorting
                    'updated_datetime': updated_datetime,
                    'categories': categories,
                    'primary_category': primary_category,
                    'comment': comment,
                    'journal_ref': journal_ref,
                    'doi': doi,
                    'entry_id': entry_id
                }
                
                papers.append(paper)
                
            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue
        
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