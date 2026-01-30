"""
Volcengine Ark API client for paper analysis, summarization, and relevance scoring.

Uses ByteDance's Doubao models via the Volcengine Ark platform.
"""
import asyncio
import os
import tempfile
from typing import Dict, List, Optional

from src.llm.base import BaseLLMClient


class ArkClient(BaseLLMClient):
    """
    LLM client using Volcengine's Ark API (ByteDance Doubao models).
    
    Supports PDF analysis (via async file upload), abstract analysis, 
    report summarization, translation, and paper relevance scoring.
    
    Note: PDF analysis uses async internally but exposes a sync interface.
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
            from volcenginesdkarkruntime import Ark, AsyncArk
            self._Ark = Ark
            self._AsyncArk = AsyncArk
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
        
        # Initialize sync client for text generation
        self.client = self._Ark(base_url=self.base_url, api_key=api_key)
        
        # Store credentials for async client (created on demand)
        self._api_key = api_key
        self._async_client = None
        
        # Model configuration
        self.text_model = self.config.get('text_model', 'doubao-seed-1-6-251015')
        self.document_model = self.config.get('document_model', 'doubao-seed-1-6-251015')
    
    def _get_async_client(self):
        """Get or create the async client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = self._AsyncArk(
                base_url=self.base_url,
                api_key=self._api_key
            )
        return self._async_client
    
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
        }
        
        # Add optional parameters if specified
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif self.max_output_tokens:
            kwargs["max_tokens"] = self.max_output_tokens
        
        # Note: Ark API may not support temperature in the same way as Gemini
        # Check documentation for supported parameters
        
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    
    async def _analyze_pdf_async(self, pdf_data: bytes, paper_metadata: Dict,
                                  prompt_type: str) -> str:
        """
        Async implementation of PDF analysis.
        
        Workflow:
        1. Write PDF bytes to temp file
        2. Upload file to Ark
        3. Wait for processing
        4. Query with file_id
        5. Cleanup
        
        Args:
            pdf_data: PDF content as bytes
            paper_metadata: Paper metadata
            prompt_type: Type of analysis
            
        Returns:
            Analysis result as text
        """
        async_client = self._get_async_client()
        temp_file_path = None
        
        try:
            # Write PDF to temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_data)
                temp_file_path = tmp.name
            
            # Upload file
            with open(temp_file_path, 'rb') as f:
                file = await async_client.files.create(
                    file=f,
                    purpose="user_data"
                )
            
            # Wait for processing
            await async_client.files.wait_for_processing(file.id)
            
            # Load prompt template
            prompt_template = self._load_prompt_template(prompt_type)
            prompt = prompt_template.format(
                title=paper_metadata['title'],
                authors=", ".join(paper_metadata['authors']),
                abstract=paper_metadata['abstract'],
                summary_length=self.summary_length
            )
            
            # Query with file reference
            response = await async_client.responses.create(
                model=self.document_model,
                input=[
                    {"role": "user", "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id
                        },
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]},
                ],
            )
            
            # Extract text from response
            # Response format may vary - handle appropriately
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'output'):
                return str(response.output)
            else:
                return str(response)
            
        finally:
            # Cleanup temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def analyze_paper_from_pdf(self, pdf_data: bytes, paper_metadata: Dict, 
                               prompt_type: str = "summary") -> str:
        """
        Analyze a paper using its PDF content and metadata.
        
        Uses async file upload internally but provides sync interface.
        
        Args:
            pdf_data: PDF content as bytes
            paper_metadata: Paper metadata (title, authors, etc.)
            prompt_type: Type of analysis to perform (summary, review, etc.)
            
        Returns:
            Analysis result as text
        """
        try:
            # Run async function synchronously
            return asyncio.run(
                self._analyze_pdf_async(pdf_data, paper_metadata, prompt_type)
            )
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

