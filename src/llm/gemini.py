"""
Google Gemini API client for paper analysis, summarization, and relevance scoring.
"""
import os
from typing import Dict, List, Optional

from src.llm.base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
    LLM client using Google's Gemini API.
    
    Supports PDF analysis, abstract analysis, report summarization,
    translation, and paper relevance scoring.
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize the Gemini client with configuration.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set
            ImportError: If google-genai package is not installed
        """
        # Initialize base class (loads config, sets common attributes)
        super().__init__(config_path)
        
        # Lazy import of Google GenAI SDK
        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
        except ImportError as e:
            raise ImportError(
                "Google GenAI SDK not installed. "
                "Install with: pip install google-genai"
            ) from e
        
        # Initialize the Google Generative AI client
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        self.client = self._genai.Client(api_key=api_key)
        self.model_name = self.config['model']
        # Note: temperature, max_output_tokens, summary_length, batch_size 
        # are already set by BaseLLMClient.__init__()
    
    def _create_generation_config(self, temperature: Optional[float] = None,
                                   max_tokens: Optional[int] = None):
        """
        Create a generation config for Gemini API calls.
        
        Args:
            temperature: Override default temperature
            max_tokens: Override default max_output_tokens
            
        Returns:
            GenerateContentConfig instance
        """
        return self._types.GenerateContentConfig(
            temperature=temperature or self.temperature,
            max_output_tokens=max_tokens or self.max_output_tokens,
            thinking_config=self._types.ThinkingConfig(thinking_budget=0)
        )
    
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
        # Load the appropriate prompt template
        prompt_template = self._load_prompt_template(prompt_type)
        
        # # Format the prompt with paper metadata
        # prompt = prompt_template.format(
        #     title=paper_metadata['title'],
        #     authors=", ".join(paper_metadata['authors']),
        #     abstract=paper_metadata['abstract'],
        #     summary_length=self.summary_length
        # )
        
        ## Updated to directly use the prompt template without wrapping in Part
        prompt = prompt_template 
        
        # Create generation config
        generation_config = self._create_generation_config()
        
        # Call the Gemini API with the PDF content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                self._types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
                prompt
            ],
            config=generation_config
        )
        
        return response.text
    
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
        # Load the appropriate prompt template
        prompt_template = self._load_prompt_template(prompt_type)
        
        # Format the prompt with paper metadata
        prompt = prompt_template.format(
            title=paper['title'],
            authors=", ".join(paper['authors']),
            abstract=paper['abstract'],
            categories=", ".join(paper['categories']),
            published_date=paper['published_date']
        )
        
        # Create generation config
        generation_config = self._create_generation_config()
        
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        
        return response.text
    
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
        # Load the report summary prompt template
        prompt_template = self._load_prompt_template("report_summary")
        
        # Prepare paper information for the prompt
        paper_info = []
        for i, paper in enumerate(papers, 1):
            paper_info.append(f"{i}. \"{paper['title']}\" by {paper['formatted_authors']}")
        
        paper_list = "\n".join(paper_info)
        
        # Format the prompt
        prompt = prompt_template.format(
            report_type=report_type,
            paper_count=len(papers),
            paper_list=paper_list,
            date=papers[0]['published_date'] if papers else "today"
        )
        
        # Create generation config
        generation_config = self._create_generation_config()
        
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        
        return response.text
    
    def translate_content(self, content: str, target_language: str) -> str:
        """
        Translate content to the target language.
        
        Args:
            content: Content to translate
            target_language: Target language code (e.g., 'zh' for Chinese)
            
        Returns:
            Translated content
        """
        # Load the translation prompt template
        prompt_template = self._load_prompt_template("translate")
        
        # Format the prompt
        prompt = prompt_template.format(
            content=content,
            target_language=target_language
        )
        
        # Create generation config with lower temperature for translation
        generation_config = self._create_generation_config(temperature=0.1)
        
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        
        return response.text
    
    def _score_single_paper(self, paper: Dict, keywords: List[str],
                            negative_keywords: Optional[List[str]] = None) -> Dict:
        """
        Score a single paper's relevance and significance using Gemini.
        
        Args:
            paper: Paper object with title, authors, abstract, etc.
            keywords: List of keywords of interest
            negative_keywords: List of keywords to avoid (optional)
            
        Returns:
            Dictionary with relevance_score, significance_score, and combined_score
        """
        # Load the relevance scoring prompt template
        prompt_template = self._load_prompt_template("relevance_scoring")
        
        # Format the prompt with paper metadata and user preferences
        prompt = prompt_template.format(
            title=paper['title'],
            authors=", ".join(paper['authors']),
            abstract=paper['abstract'],
            categories=", ".join(paper['categories']),
            published_date=paper['published_date'],
            keywords=", ".join(keywords),
            negative_keywords=", ".join(negative_keywords or [])
        )
        
        # Create generation config with lower temperature for more consistent scoring
        generation_config = self._create_generation_config(
            temperature=0.05,  # Very low for stable evaluation
            max_tokens=1024
        )
        
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        
        return self._parse_json_response(response.text)
    
    # Backward compatibility alias
    def score_paper_relevance(self, paper: Dict, keywords: List[str],
                              negative_keywords: Optional[List[str]] = None) -> Dict:
        """
        Alias for _score_single_paper for backward compatibility.
        
        Deprecated: Use batch_score_papers() instead.
        """
        return self._score_single_paper(paper, keywords, negative_keywords)
