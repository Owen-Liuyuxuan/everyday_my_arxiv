"""
Base class for LLM clients providing abstract interface for paper analysis.
"""
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Provides common functionality for loading prompts and batch processing,
    while requiring subclasses to implement provider-specific API calls.
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize the LLM client with configuration.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)['llm']
        
        # Common configuration attributes
        self.temperature = self.config.get('temperature', 0.2)
        self.max_output_tokens = self.config.get('max_output_tokens', 4096)
        self.summary_length = self.config.get('summary_length', 'medium')
        self.batch_size = self.config.get('batch_size', 16)
    
    def _load_prompt_template(self, prompt_name: str) -> str:
        """
        Load a prompt template from file.
        
        Args:
            prompt_name: Name of the prompt template (without .txt extension)
            
        Returns:
            The prompt template content as a string
        """
        prompt_path = f"src/llm/prompts/{prompt_name}.txt"
        with open(prompt_path, 'r') as f:
            return f.read()
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON from LLM response text.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dict or default scores on failure
        """
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return {"error": "No valid JSON found in response", "raw_response": response_text}
        except json.JSONDecodeError:
            return {
                "relevance_score": 1,
                "significance_score": 1,
                "combined_score": 2
            }
    
    @abstractmethod
    def analyze_paper_from_pdf(self, pdf_data: bytes, paper_metadata: Dict, 
                               prompt_type: str = "summary") -> str:
        """
        Analyze a paper using its PDF content and metadata.
        
        Args:
            pdf_data: PDF content as bytes
            paper_metadata: Paper metadata (title, authors, etc.)
            prompt_type: Type of analysis to perform (summary, review, etc.)
            
        Returns:
            Analysis result as text
        """
        pass
    
    @abstractmethod
    def analyze_paper_from_abstract(self, paper: Dict, 
                                    prompt_type: str = "abstract_analysis") -> str:
        """
        Analyze a paper using only its abstract and metadata.
        
        Args:
            paper: Paper object with title, authors, abstract, etc.
            prompt_type: Type of analysis to perform
            
        Returns:
            Analysis result as text
        """
        pass
    
    @abstractmethod
    def generate_report_summary(self, papers: List[Dict], 
                                report_type: str = "daily") -> str:
        """
        Generate a summary of multiple papers for the report.
        
        Args:
            papers: List of paper objects
            report_type: Type of report (daily, weekly, etc.)
            
        Returns:
            Report summary as text
        """
        pass
    
    @abstractmethod
    def translate_content(self, content: str, target_language: str) -> str:
        """
        Translate content to the target language.
        
        Args:
            content: Content to translate
            target_language: Target language code (e.g., 'zh' for Chinese)
            
        Returns:
            Translated content
        """
        pass
    
    @abstractmethod
    def _score_single_paper(self, paper: Dict, keywords: List[str],
                            negative_keywords: Optional[List[str]] = None,
                            author_preferences: Optional[Dict] = None) -> Dict:
        """
        Score a single paper's relevance and significance.

        This is the provider-specific implementation called by batch_score_papers.

        Args:
            paper: Paper object with title, authors, abstract, etc.
            keywords: List of keywords of interest
            negative_keywords: List of keywords to avoid (optional)
            author_preferences: Dict of preferred authors/institutions (optional)

        Returns:
            Dictionary with relevance_score, significance_score, and combined_score
        """
        pass

    def batch_score_papers(self, papers: List[Dict], keywords: List[str],
                          negative_keywords: Optional[List[str]] = None,
                          author_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Score multiple papers in batches for efficiency.

        This is the public API that uses _score_single_paper internally.

        Args:
            papers: List of paper objects
            keywords: List of keywords of interest
            negative_keywords: List of keywords to avoid (optional)
            author_preferences: Dict of preferred authors/institutions (optional)

        Returns:
            List of papers with added relevance and significance scores
        """
        scored_papers = []

        # Process papers in batches
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i:i + self.batch_size]
            print(f"Scoring batch {i//self.batch_size + 1}/{(len(papers)-1)//self.batch_size + 1} " +
                  f"({len(batch)} papers)")

            # Process each paper in the batch
            for paper in batch:
                scores = self._score_single_paper(paper, keywords, negative_keywords, author_preferences)
                
                # Add scores to the paper object
                paper['relevance_score'] = scores.get('relevance_score', 1)
                paper['significance_score'] = scores.get('significance_score', 1)
                paper['combined_score'] = scores.get('combined_score', 2)
                
                scored_papers.append(paper)
                
        return scored_papers

