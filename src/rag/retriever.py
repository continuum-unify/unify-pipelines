from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import time

from src.embedding import embedding_client
from src.milvus_client import milvus_client
from src.logs.logger import setup_logger
from .schemas import ArxivSource

logger = setup_logger(__name__)

class BaseRetriever(ABC):
    """Abstract base class for document retrieval."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[ArxivSource]:
        """Retrieve relevant documents for the given query."""
        pass

class MilvusRetriever(BaseRetriever):
    """Milvus-based vector similarity search retriever."""
    
    def __init__(
        self,
        embedding_client=embedding_client,
        milvus_client=milvus_client,
        search_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the retriever with clients and search parameters.
        
        Args:
            embedding_client: Client for generating embeddings
            milvus_client: Client for Milvus operations
            search_params: Optional search parameters override
        """
        self.embedding_client = embedding_client
        self.milvus_client = milvus_client
        
        # Default search parameters
        self.search_params = search_params or {
            "metric_type": "L2",
            "params": {"search_width": 128},
        }
        
        # Fields to retrieve from Milvus
        self.output_fields = [
            "doc_id", "arxiv_url_link", "source_file", "year",
            "category", "abstract", "text", "summary", "key_points",
            "technical_terms", "relationships", "timestamp"
        ]

    def verify_collection(self) -> bool:
        """Verify Milvus collection setup and indexing."""
        try:
            collection = self.milvus_client.get_collection()
            if not collection:
                logger.error("Failed to get collection")
                return False

            # Log collection info
            logger.info(f"Collection name: {collection.name}")
            logger.info(f"Number of entities: {collection.num_entities}")
            
            # Verify collection has data
            if collection.is_empty:
                logger.warning("Collection is empty")
                return False

            # Check indexes
            indexes = collection.indexes
            logger.info(f"Available indexes: {indexes}")
            
            # Verify vector index exists
            has_vector_index = any(idx.field_name == "embedding" for idx in indexes)
            if not has_vector_index:
                logger.error("No index found on embedding field")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying collection: {str(e)}")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[ArxivSource]:
        """Retrieve relevant documents using vector similarity search.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            
        Returns:
            List of ArxivSource objects sorted by relevance
        """
        try:
            # Generate query embedding
            start_time = time.time()
            query_embedding = self.embedding_client.get_embedding(query, input_type="query")
            if not query_embedding:
                raise ValueError("Failed to generate query embedding")
            
            logger.debug(f"Generated embedding in {time.time() - start_time:.2f}s")

            # Connect to Milvus
            if not self.milvus_client.connect():
                raise ConnectionError("Failed to connect to Milvus")

            # Get and verify collection
            collection = self.milvus_client.get_collection()
            if not collection:
                raise ValueError("Failed to get collection")

            # Prepare search parameters
            search_params = {
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": self.search_params,
                "limit": top_k,
                "output_fields": self.output_fields
            }

            # Execute search
            logger.debug(f"Executing search with params: {search_params}")
            search_start = time.time()
            results = collection.search(**search_params)
            logger.debug(f"Search completed in {time.time() - search_start:.2f}s")

            if not results or not results[0]:
                logger.warning("No search results found")
                return []

            # Process results
            sources = []
            for hit in results[0]:
                try:
                    source = ArxivSource(
                        doc_id=getattr(hit, 'doc_id', -1),
                        arxiv_url_link=getattr(hit, 'arxiv_url_link', ''),
                        source_file=getattr(hit, 'source_file', ''),
                        year=getattr(hit, 'year', 0),
                        category=getattr(hit, 'category', ''),
                        abstract=getattr(hit, 'abstract', ''),
                        text=getattr(hit, 'text', ''),
                        summary=getattr(hit, 'summary', ''),
                        key_points=getattr(hit, 'key_points', ''),
                        technical_terms=getattr(hit, 'technical_terms', ''),
                        relationships=getattr(hit, 'relationships', ''),
                        timestamp=getattr(hit, 'timestamp', 0),
                        score=float(hit.distance)
                    )
                    sources.append(source)
                    logger.debug(f"Processed hit for doc_id: {source.doc_id}, score: {source.score:.4f}")
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
                    continue

            logger.info(f"Retrieved {len(sources)} documents for query: {query}")
            return sources

        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            raise
        finally:
            if self.milvus_client:
                self.milvus_client.release_collection()
                self.milvus_client.disconnect()

    def estimate_search_quality(self, sources: List[ArxivSource]) -> Dict[str, float]:
        """Estimate search quality metrics.
        
        Args:
            sources: List of retrieved sources
            
        Returns:
            Dictionary containing quality metrics
        """
        if not sources:
            return {
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "score_variance": 0.0
            }

        scores = [s.score for s in sources]
        avg_score = sum(scores) / len(scores)
        
        return {
            "avg_score": avg_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "score_variance": sum((s - avg_score) ** 2 for s in scores) / len(scores)
        }

class HybridRetriever(BaseRetriever):
    """Combines multiple retrieval methods for better results."""
    
    def __init__(
        self,
        vector_retriever: MilvusRetriever,
        reranker: Optional[Any] = None,  # Will implement reranker later
        combine_method: str = "merge"
    ):
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self.combine_method = combine_method

    def retrieve(self, query: str, top_k: int = 5) -> List[ArxivSource]:
        """
        Retrieve documents using multiple methods and combine results.
        
        Currently implements:
        1. Vector similarity search
        2. Optional reranking of results
        
        Future additions could include:
        - BM25 text search
        - Filter-based retrieval
        - Ensemble methods
        """
        # Get initial results from vector search
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k)
        
        # If we have a reranker, apply it
        if self.reranker and vector_results:
            try:
                reranked_results = self.reranker.rerank(query, vector_results)
                logger.info("Successfully reranked results")
                return reranked_results
            except Exception as e:
                logger.error(f"Reranking failed: {str(e)}")
                return vector_results
        
        return vector_results

# Factory function to create appropriate retriever
def create_retriever(
    retriever_type: str = "milvus",
    search_params: Optional[Dict[str, Any]] = None,
    reranker: Optional[Any] = None
) -> BaseRetriever:
    """Create a retriever instance based on specified type."""
    if retriever_type == "milvus":
        return MilvusRetriever(search_params=search_params)
    elif retriever_type == "hybrid":
        vector_retriever = MilvusRetriever(search_params=search_params)
        return HybridRetriever(vector_retriever, reranker)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")