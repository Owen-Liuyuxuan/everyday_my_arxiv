"""
Google Gemini API client for paper analysis, summarization, and relevance scoring.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import re

class GeminiClient:
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the Gemini client with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)['llm']
        
        # Initialize the Google Generative AI client
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        self.client = genai.Client(api_key=api_key)
        
        self.model_name = self.config['model']
        self.temperature = self.config['temperature']
        self.max_output_tokens = self.config['max_output_tokens']
        self.summary_length = self.config['summary_length']
        self.batch_size = self.config.get('batch_size', 16)  # Default batch size if not specified
    
    def _load_prompt_template(self, prompt_name: str) -> str:
        """Load a prompt template from file."""
        prompt_path = f"src/llm/prompts/{prompt_name}.txt"
        with open(prompt_path, 'r') as f:
            return f.read()
    
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
        
        # Format the prompt with paper metadata
        prompt = prompt_template.format(
            title=paper_metadata['title'],
            authors=", ".join(paper_metadata['authors']),
            abstract=paper_metadata['abstract'],
            summary_length=self.summary_length
        )
        
        # Create generation config
        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        
        # Call the Gemini API with the PDF content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
                prompt
            ],
            config=generation_config
        )
        
        return response.text
    
    def analyze_paper_from_abstract(self, paper: Dict, prompt_type: str = "abstract_analysis") -> str:
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
        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        
        return response.text
    
    def generate_report_summary(self, papers: List[Dict], report_type: str = "daily") -> str:
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
        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        
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
        generation_config = types.GenerateContentConfig(
            temperature=0.1,  # Lower temperature for more accurate translation
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        
        return response.text
        
    def score_paper_relevance(self, paper: Dict, keywords: List[str], 
                             negative_keywords: List[str] = None) -> Dict:
        """
        Score a paper's relevance and significance based on metadata and user preferences.
        
        Args:
            paper: Paper object with title, authors, abstract, etc.
            keywords: List of keywords of interest
            negative_keywords: List of keywords to avoid (optional)
            
        Returns:
            Dictionary with relevance and significance scores
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
        generation_config = types.GenerateContentConfig(
            temperature=0.05, # for very accurate and stable evaluation
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        
        try:
            # Parse the JSON response
            json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            else:
                return {"error": "No valid JSON found in scoring", "raw_response": response}
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            return {
                "relevance_score": 1,
                "significance_score": 1,
                "combined_score": 2
            }
    
    def batch_score_papers(self, papers: List[Dict], keywords: List[str], 
                          negative_keywords: List[str] = None) -> List[Dict]:
        """
        Score multiple papers in batches for efficiency.
        
        Args:
            papers: List of paper objects
            keywords: List of keywords of interest
            negative_keywords: List of keywords to avoid (optional)
            
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
                scores = self.score_paper_relevance(paper, keywords, negative_keywords)
                
                # Add scores to the paper object
                paper['relevance_score'] = scores.get('relevance_score', 1)
                paper['significance_score'] = scores.get('significance_score', 1)
                paper['combined_score'] = scores.get('combined_score', 2)
                
                scored_papers.append(paper)
                
        return scored_papers