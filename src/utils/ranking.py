"""
Paper ranking and selection utilities.
"""
from typing import Dict, List, Optional, Tuple

class PaperRanker:
    def __init__(self, min_combined_score: int = 4):
        """
        Initialize the paper ranker.
        
        Args:
            min_combined_score: Minimum combined score (relevance + significance) to consider a paper
        """
        self.min_combined_score = min_combined_score
    
    def rank_papers_by_scores(self, papers: List[Dict]) -> List[Dict]:
        """
        Rank papers by their combined relevance and significance scores.
        
        Args:
            papers: List of paper objects with relevance and significance scores
            
        Returns:
            List of papers sorted by combined score (descending)
        """
        # Sort papers by combined score (descending)
        return sorted(papers, key=lambda p: p.get('combined_score', 0), reverse=True)
    
    def filter_by_minimum_score(self, papers: List[Dict]) -> List[Dict]:
        """
        Filter papers by minimum combined score.
        
        Args:
            papers: List of paper objects with combined scores
            
        Returns:
            List of papers with combined score >= min_combined_score
        """
        return [p for p in papers if p.get('combined_score', 0) >= self.min_combined_score]
    
    def select_top_papers(self, papers: List[Dict], limit: int = 10) -> List[Dict]:
        """
        Select top papers based on combined score.
        
        Args:
            papers: List of paper objects with combined scores
            limit: Maximum number of papers to select
            
        Returns:
            List of top papers
        """
        # Filter by minimum score
        filtered_papers = self.filter_by_minimum_score(papers)
        
        # Rank papers
        ranked_papers = self.rank_papers_by_scores(filtered_papers)
        
        # Return top papers
        return ranked_papers[:limit]
    
    def get_selection_stats(self, papers: List[Dict]) -> Dict:
        """
        Get statistics about the paper selection.
        
        Args:
            papers: List of paper objects with scores
            
        Returns:
            Dictionary with selection statistics
        """
        if not papers:
            return {
                "total": 0,
                "avg_relevance": 0,
                "avg_significance": 0,
                "avg_combined": 0,
                "high_relevance_count": 0,
                "high_significance_count": 0
            }
        
        total = len(papers)
        avg_relevance = sum(p.get('relevance_score', 0) for p in papers) / total
        avg_significance = sum(p.get('significance_score', 0) for p in papers) / total
        avg_combined = sum(p.get('combined_score', 0) for p in papers) / total
        high_relevance_count = sum(1 for p in papers if p.get('relevance_score', 0) == 3)
        high_significance_count = sum(1 for p in papers if p.get('significance_score', 0) == 3)
        
        return {
            "total": total,
            "avg_relevance": avg_relevance,
            "avg_significance": avg_significance,
            "avg_combined": avg_combined,
            "high_relevance_count": high_relevance_count,
            "high_significance_count": high_significance_count
        }