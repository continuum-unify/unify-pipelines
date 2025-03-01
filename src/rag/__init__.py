"""
RAG (Retrieval-Augmented Generation) package for academic paper analysis.
Provides components for vector search, context formatting, and LLM response generation.
"""

from .engine import RAGEngine, create_rag_engine
from .retriever import BaseRetriever, MilvusRetriever, HybridRetriever, create_retriever
from .formatter import BaseFormatter, AcademicFormatter, CustomFormatter, create_formatter
from .schemas import (
    ArxivSource,
    RAGResponse,
    SearchMetrics,
    ModelMetrics,
    create_search_metrics,
    create_model_metrics
)

# Version info
__version__ = '0.1.0'
__author__ = 'Your Name'

# Default factory function for easy initialization
def create_academic_rag(config: dict = None) -> RAGEngine:
    """Create a RAG engine configured for academic paper analysis.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RAGEngine instance
    """
    return create_rag_engine(
        config=config,
        retriever_type='milvus',
        formatter_type='academic'
    )

__all__ = [
    # Main engine
    'RAGEngine',
    'create_rag_engine',
    'create_academic_rag',
    
    # Retrievers
    'BaseRetriever',
    'MilvusRetriever',
    'HybridRetriever',
    'create_retriever',
    
    # Formatters
    'BaseFormatter',
    'AcademicFormatter',
    'CustomFormatter',
    'create_formatter',
    
    # Data models
    'ArxivSource',
    'RAGResponse',
    'SearchMetrics',
    'ModelMetrics',
    'create_search_metrics',
    'create_model_metrics'
]