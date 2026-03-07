"""
OpenAI-compatible API client for paper analysis and relevance scoring.

Supports any OpenAI-compatible API endpoint including:
- OpenAI models (GPT-4, GPT-4o, etc.)
- DeepSeek models
- Local models via Ollama, LM Studio, etc.
- Other OpenAI-compatible providers (Anthropic via compatible endpoints, etc.)

Note: PDF analysis is NOT supported by this client as it requires multimodal capabilities.
Use a separate multimodal client (GeminiClient or ArkClient) for PDF analysis.
"""
import os
from typing import Dict, List, Optional

from src.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    LLM client using OpenAI-compatible API.
    
    Supports text-based operations:
    - Abstract analysis
    - Paper relevance scoring
    - Report summarization
    - Translation
    
    PDF analysis is NOT supported - use a separate multimodal client.
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize the OpenAI-compatible client with configuration.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
            ImportError: If openai package is not installed
        """
        # Initialize base class (loads config, sets common attributes)
        super().__init__(config_path)
        
        # Lazy import of OpenAI SDK
        try:
            from openai import OpenAI
            self._OpenAI = OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK not installed. "
                "Install with: pip install openai"
            ) from e
        
        # Get API key from environment or config
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try to get from config
            api_key = self.config.get('api_key')
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set, "
                "or 'api_key' not found in config"
            )
        
        # Get base URL from config (for OpenAI-compatible endpoints)
        base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        
        # Initialize the OpenAI client
        self.client = self._OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Model configuration
        self.model_name = self.config.get('model', 'gpt-4o-mini')
    
    def _call_api(self, prompt: str, temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> str:
        """
        Make a synchronous text generation API call.
        
        Args:
            prompt: The prompt text
            temperature: Override default temperature
            max_tokens: Override default max_output_tokens
            
        Returns:
            Generated text response
        """
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self.temperature:
            kwargs["temperature"] = self.temperature
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif self.max_output_tokens:
            kwargs["max_tokens"] = self.max_output_tokens
        
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    
    def analyze_paper_from_pdf(self, pdf_data: bytes, paper_metadata: Dict,
                               prompt_type: str = "summary") -> str:
        """
        Analyze a paper using its PDF content and metadata.
        
        NOTE: OpenAI-compatible clients typically don't support PDF analysis.
        This method will raise a NotImplementedError.
        
        Use a separate multimodal client (GeminiClient or ArkClient) for PDF analysis.
        
        Args:
            pdf_data: PDF content as bytes (ignored)
            paper_metadata: Paper metadata (title, authors, etc.)
            prompt_type: Type of analysis to perform
            
        Returns:
            Analysis result as text
            
        Raises:
            NotImplementedError: PDF analysis not supported by OpenAI-compatible client
        """
        raise NotImplementedError(
            "PDF analysis is not supported by OpenAI-compatible client. "
            "Use GeminiClient or ArkClient for PDF analysis, or configure "
            "separate pdf_provider in your config."
        )
    
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
        
        return self._call_api(prompt)
    
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
        
        return self._call_api(prompt)
    
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
        
        # Use lower temperature for translation
        return self._call_api(prompt, temperature=0.1)
    
    def _score_single_paper(self, paper: Dict, keywords: List[str],
                            negative_keywords: Optional[List[str]] = None,
                            author_preferences: Optional[Dict] = None) -> Dict:
        """
        Score a single paper's relevance and significance using OpenAI-compatible API.

        Args:
            paper: Paper object with title, authors, abstract, etc.
            keywords: List of keywords of interest
            negative_keywords: List of keywords to avoid (optional)
            author_preferences: Dict of preferred authors/institutions (optional)

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
            categories=", ".join(paper.get('categories', [])),
            published_date=paper.get('published_date', 'N/A'),
            venue=paper.get('venue', 'N/A'),
            code_url=paper.get('code_url', 'N/A'),
            keywords=", ".join(keywords),
            negative_keywords=", ".join(negative_keywords or []),
            author_preferences=self._format_author_preferences(author_preferences)
        )

        response_text = self._call_api(prompt, temperature=0.05, max_tokens=1024)
        return self._parse_json_response(response_text)

    def _format_author_preferences(self, author_preferences: Optional[Dict]) -> str:
        """Format author preferences for the prompt."""
        if not author_preferences:
            return "No specific author preferences"

        formatted = []
        for category, values in author_preferences.items():
            if values:
                formatted.append(f"{category}: {', '.join(values)}")
        return "; ".join(formatted) if formatted else "No specific author preferences"
