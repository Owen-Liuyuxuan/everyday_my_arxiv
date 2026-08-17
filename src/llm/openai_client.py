"""
OpenAI API client for paper analysis and relevance scoring.

Supports any OpenAI-compatible API endpoint including:
- OpenAI models (GPT-4, GPT-4o, etc.)
- DeepSeek models
- Local models via Ollama, LM Studio, etc.
- Other OpenAI-compatible providers (Anthropic via compatible endpoints, etc.)

Official OpenAI endpoints also support PDF inputs through the Responses API. Other
OpenAI-compatible providers may not implement the Files or Responses APIs.
"""
import os
import re
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
    
    PDF analysis uses the Files and Responses APIs when the configured endpoint
    implements those official OpenAI APIs.
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
        self.model_name = self.config.get('model', 'gpt-5.6-luna')
        self.pdf_detail = self.config.get('pdf_detail', 'auto')
        if self.pdf_detail not in {'auto', 'low', 'high'}:
            raise ValueError("llm.pdf_detail must be one of: auto, low, high")

        self.delete_uploaded_files = self.config.get('delete_uploaded_files', True)
    
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
        
        if self.model_name.lower().startswith("gpt-"):
            # OpenAI GPT reasoning models only accept the default temperature.
            # Keep custom temperatures for non-GPT OpenAI-compatible models.
            kwargs["temperature"] = 1
        elif temperature is not None:
            kwargs["temperature"] = temperature
        elif self.temperature:
            kwargs["temperature"] = self.temperature
        
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        elif self.max_output_tokens:
            kwargs["max_completion_tokens"] = self.max_output_tokens
        
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    
    def analyze_paper_from_pdf(self, pdf_data: bytes, paper_metadata: Dict,
                               prompt_type: str = "summary") -> str:
        """
        Analyze a PDF with the OpenAI Files and Responses APIs.

        The uploaded file is deleted after the response by default. Set
        ``llm.delete_uploaded_files`` to ``false`` to retain it.
        
        Args:
            pdf_data: PDF content as bytes
            paper_metadata: Paper metadata (title, authors, etc.)
            prompt_type: Type of analysis to perform
            
        Returns:
            Analysis result as text
            
        Raises:
            ValueError: If the input is empty, not a PDF, or exceeds the API limit
        """
        if not pdf_data:
            raise ValueError("PDF data is empty")
        if not pdf_data.lstrip().startswith(b"%PDF-"):
            raise ValueError("Input does not appear to be a PDF (missing %PDF header)")

        # OpenAI's combined per-request file-input limit is 50 MB. Keep the
        # validation strict because the documentation says each file must be under it.
        max_pdf_bytes = 50 * 1024 * 1024
        if len(pdf_data) >= max_pdf_bytes:
            raise ValueError("PDF must be smaller than 50 MB")

        prompt = self._load_prompt_template(prompt_type)
        title = str(paper_metadata.get('title', 'paper')).strip() or 'paper'
        safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', title)[:80].strip('._') or 'paper'
        uploaded_file = None

        try:
            uploaded_file = self.client.files.create(
                file=(f"{safe_stem}.pdf", pdf_data, "application/pdf"),
                purpose="user_data",
            )

            input_file = {
                "type": "input_file",
                "file_id": uploaded_file.id,
            }
            # In SDK 2.26.0, omitting detail represents the API's `auto` default.
            if self.pdf_detail in {'low', 'high'}:
                input_file["detail"] = self.pdf_detail

            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            input_file,
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
                max_output_tokens=self.max_output_tokens,
            )
            return response.output_text
        finally:
            if uploaded_file is not None and self.delete_uploaded_files:
                try:
                    self.client.files.delete(uploaded_file.id)
                except Exception as exc:
                    # Cleanup failure should not discard a successful analysis.
                    print(f"Warning: could not delete OpenAI file {uploaded_file.id}: {exc}")
    
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
