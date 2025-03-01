from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class ArxivSource:
    """Represents an academic paper source from the arxiv_documents collection.
    
    Attributes:
        doc_id (int): Unique identifier for the document
        arxiv_url_link (str): Direct link to the arXiv paper
        source_file (str): Original source filename
        year (int): Publication year
        category (str): Paper category/subject area
        abstract (str): Paper abstract
        text (str): Main text content
        summary (str): Generated summary
        key_points (str): Extracted key points
        technical_terms (str): Technical terminology
        relationships (str): Identified relationships
        timestamp (int): Processing timestamp
        score (float): Search relevance score
    """
    doc_id: int
    arxiv_url_link: str
    source_file: str
    year: int
    category: str
    abstract: str
    text: str
    summary: str
    key_points: str
    technical_terms: str
    relationships: str
    timestamp: int
    score: float

    @property
    def formatted_date(self) -> str:
        """Returns human-readable timestamp"""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')

    @property
    def title(self) -> str:
        """Extracts paper title from source file name"""
        clean_name = self.source_file.replace('_embedded.json', '')
        if len(clean_name) > 4 and clean_name[:4].isdigit():
            return clean_name[5:].replace('-', ' ').strip()
        return clean_name.replace('-', ' ').strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert source to dictionary format"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "year": self.year,
            "category": self.category,
            "url": self.arxiv_url_link,
            "abstract": self.abstract,
            "key_points": self.key_points,
            "technical_terms": self.technical_terms,
            "score": self.score
        }

@dataclass
class SearchMetrics:
    """Tracks performance metrics for the search operation.
    
    Attributes:
        start_time (float): Search operation start time
        embedding_time (float): Time taken for embedding generation
        search_time (float): Time taken for vector search
        total_time (float): Total operation time
        num_results (int): Number of results found
        avg_score (float): Average relevance score
    """
    start_time: float
    embedding_time: float
    search_time: float
    total_time: float
    num_results: int
    avg_score: float

@dataclass
class ModelMetrics:
    """Tracks performance metrics for the model operation.
    
    Attributes:
        start_time (float): Model operation start time
        context_time (float): Time taken for context preparation
        inference_time (float): Time taken for model inference
        total_time (float): Total operation time
        prompt_tokens (int): Number of input tokens
        completion_tokens (int): Number of output tokens
    """
    start_time: float
    context_time: float
    inference_time: float
    total_time: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline.
    
    Attributes:
        question (str): Original research question
        answer (str): Generated answer
        sources (List[ArxivSource]): Retrieved source documents
        search_metrics (SearchMetrics): Search performance metrics
        model_metrics (ModelMetrics): Model performance metrics
        error (Optional[str]): Error message if any
    """
    question: str
    answer: str
    sources: List[ArxivSource]
    search_metrics: SearchMetrics
    model_metrics: ModelMetrics
    error: Optional[str] = None

    @property
    def total_time(self) -> float:
        """Total processing time"""
        return self.search_metrics.total_time + self.model_metrics.total_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "metrics": {
                "search": {
                    "time": self.search_metrics.total_time,
                    "results": self.search_metrics.num_results,
                    "avg_score": self.search_metrics.avg_score
                },
                "model": {
                    "time": self.model_metrics.total_time,
                    "prompt_tokens": self.model_metrics.prompt_tokens,
                    "completion_tokens": self.model_metrics.completion_tokens
                },
                "total_time": self.total_time
            },
            "error": self.error
        }

# Helper functions for creating metrics objects
def create_search_metrics(
    start_time: float,
    embedding_time: float,
    search_time: float,
    sources: List[ArxivSource]
) -> SearchMetrics:
    """Creates SearchMetrics from timing data and sources"""
    return SearchMetrics(
        start_time=start_time,
        embedding_time=embedding_time,
        search_time=search_time,
        total_time=time() - start_time,
        num_results=len(sources),
        avg_score=sum(s.score for s in sources) / len(sources) if sources else 0.0
    )

def create_model_metrics(
    start_time: float,
    context_time: float,
    inference_time: float,
    response: Optional[Dict[str, Any]] = None
) -> ModelMetrics:
    """Creates ModelMetrics from timing data and optional response info"""
    return ModelMetrics(
        start_time=start_time,
        context_time=context_time,
        inference_time=inference_time,
        total_time=time() - start_time,
        prompt_tokens=response.get('usage', {}).get('prompt_tokens') if response else None,
        completion_tokens=response.get('usage', {}).get('completion_tokens') if response else None
    )