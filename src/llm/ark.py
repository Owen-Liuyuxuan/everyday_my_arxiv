"""
Volcengine Ark API client for paper analysis, summarization, and relevance scoring.

Uses ByteDance's Doubao models via the Volcengine Ark platform.
"""
import base64
import os
from typing import Dict, List, Optional

from src.llm.base import BaseLLMClient


class ArkClient(BaseLLMClient):
    """
    LLM client using Volcengine's Ark API (ByteDance Doubao models).
    
    Supports PDF analysis (via base64 encoding), abstract analysis, 
    report summarization, translation, and paper relevance scoring.
    """
    
    def __init__(self, config_path: str = "config/config_ark.json"):
        """
        Initialize the Ark client with configuration.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ValueError: If ARK_API_KEY environment variable is not set
            ImportError: If volcengine SDK is not installed
        """
        # Initialize base class (loads config, sets common attributes)
        super().__init__(config_path)
        
        # Lazy import of Volcengine Ark SDK
        try:
            from volcenginesdkarkruntime import Ark
            self._Ark = Ark
        except ImportError as e:
            raise ImportError(
                "Volcengine Ark SDK not installed. "
                "Install with: pip install 'volcengine-python-sdk[ark]'"
            ) from e
        
        # Get API key from environment
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError("ARK_API_KEY environment variable is not set")
        
        # Get base URL from config or use default
        self.base_url = self.config.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3')
        
        # Initialize sync client
        self.client = self._Ark(base_url=self.base_url, api_key=api_key)
        
        # Model configuration
        self.text_model = self.config.get('text_model', 'doubao-seed-1-6-251015')
        self.document_model = self.config.get('document_model', 'doubao-seed-1-6-251015')
    
    def _call_text_api(self, prompt: str, temperature: Optional[float] = None,
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
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.text_model,
            "messages": messages,
            "thinking": {"type": "disabled"},  # Disable deep thinking for faster response
        }
        
        # Add optional parameters if specified
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif self.max_output_tokens:
            kwargs["max_tokens"] = self.max_output_tokens
        
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    
    def _call_document_api(self, pdf_data: bytes, prompt: str,
                           max_tokens: Optional[int] = None) -> str:
        """
        Make a synchronous document analysis API call using base64 encoding.
        
        Args:
            pdf_data: PDF content as bytes
            prompt: The prompt text
            max_tokens: Override default max_output_tokens
            
        Returns:
            Generated text response
        """
        # Encode PDF to base64
        base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        
        kwargs = {
            "model": self.document_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_data": f"data:application/pdf;base64,{base64_pdf}",
                            "filename": "paper.pdf"
                        },
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "thinking": {"type": "disabled"},  # Disable deep thinking for faster response
        }
        
        # Set max_output_tokens with override support (consistent with _call_text_api)
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        elif self.max_output_tokens:
            kwargs["max_output_tokens"] = self.max_output_tokens
        
        response = self.client.responses.create(**kwargs)
        
        # Extract text from response
        # Response structure: response.output[0].content[0].text
        # where output is a list of ResponseOutputMessage objects
        if hasattr(response, 'output') and isinstance(response.output, list):
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'message':
                    if hasattr(item, 'content') and isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                if hasattr(content_item, 'text') and content_item.text:
                                    return content_item.text
        
        # Fallback: try other common patterns
        if hasattr(response, 'text') and response.text:
            return response.text
        
        # Last resort: stringify the response
        return str(response)
    
    def analyze_paper_from_pdf(self, pdf_data: bytes, paper_metadata: Dict, 
                               prompt_type: str = "summary") -> str:
        """
        Analyze a paper using its PDF content and metadata.
        
        Uses base64 encoding for synchronous PDF analysis.
        
        Args:
            pdf_data: PDF content as bytes
            paper_metadata: Paper metadata (title, authors, etc.)
            prompt_type: Type of analysis to perform (summary, review, etc.)
            
        Returns:
            Analysis result as text
        """
        try:
            # Load prompt template
            prompt_template = self._load_prompt_template(prompt_type)
            prompt = prompt_template.format(
                title=paper_metadata['title'],
                authors=", ".join(paper_metadata['authors']),
                abstract=paper_metadata['abstract'],
                summary_length=self.summary_length
            )
            
            return self._call_document_api(pdf_data, prompt)
            
        except Exception as e:
            print(f"Error analyzing PDF with Ark: {e}")
            # Fall back to abstract analysis
            print("Falling back to abstract-based analysis")
            return self.analyze_paper_from_abstract(paper_metadata, prompt_type="abstract_analysis")
    
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
        
        return self._call_text_api(prompt)
    
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
        
        return self._call_text_api(prompt)
    
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
        
        return self._call_text_api(prompt)
    
    def _score_single_paper(self, paper: Dict, keywords: List[str],
                            negative_keywords: Optional[List[str]] = None) -> Dict:
        """
        Score a single paper's relevance and significance using Ark.
        
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
        
        response_text = self._call_text_api(prompt, max_tokens=1024)
        return self._parse_json_response(response_text)
