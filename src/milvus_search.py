from typing import List, Dict, Any, Optional
import time
import numpy as np
from pymilvus import Collection, AnnSearchRequest, RRFRanker, WeightedRanker

class MilvusSearchTester:
    def __init__(self, collection: Collection):
        """Initialize the search tester with a Milvus collection."""
        self.collection = collection
        self.search_latency_fmt = "Search latency = {:.4f}s"
        
    def basic_vector_search(
        self, 
        query_vector: np.ndarray,
        limit: int = 10,
        output_fields: List[str] = ["arxiv_url_link", "abstract", "year", "category"]
    ) -> List[Dict]:
        """
        Perform basic vector similarity search.
        
        Args:
            query_vector: The query embedding vector
            limit: Number of results to return
            output_fields: Fields to include in results
        """
        start_time = time.time()
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=output_fields
        )
        
        end_time = time.time()
        print(self.search_latency_fmt.format(end_time - start_time))
        
        return self._format_results(results[0])

    def hybrid_category_search(
        self,
        query_vector: np.ndarray,
        category: str,
        year_range: Optional[tuple] = None,
        limit: int = 10,
        output_fields: List[str] = ["arxiv_url_link", "abstract", "year", "category"]
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity with category and year filtering.
        
        Args:
            query_vector: The query embedding vector
            category: Category to filter by
            year_range: Optional tuple of (start_year, end_year)
            limit: Number of results to return
            output_fields: Fields to include in results
        """
        start_time = time.time()
        
        # Construct expression
        expr = f'category == "{category}"'
        if year_range:
            expr += f' and year >= {year_range[0]} and year <= {year_range[1]}'
            
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            expr=expr,
            limit=limit,
            output_fields=output_fields
        )
        
        end_time = time.time()
        print(self.search_latency_fmt.format(end_time - start_time))
        
        return self._format_results(results[0])

    def multi_vector_search(
        self,
        query_vectors: Dict[str, np.ndarray],
        weights: Optional[List[float]] = None,
        limit: int = 10,
        output_fields: List[str] = ["arxiv_url_link", "abstract", "year", "category"]
    ) -> List[Dict]:
        """
        Perform multi-vector search using multiple embeddings with optional weights.
        
        Args:
            query_vectors: Dictionary of field names and their query vectors
            weights: Optional list of weights for each vector field
            limit: Number of results to return
            output_fields: Fields to include in results
        """
        start_time = time.time()
        
        search_reqs = []
        for field_name, vector in query_vectors.items():
            search_param = {
                "data": [vector],
                "anns_field": field_name,
                "param": {"metric_type": "L2", "params": {"nprobe": 16}},
                "limit": limit
            }
            search_reqs.append(AnnSearchRequest(**search_param))
        
        # Use weighted ranker if weights provided, otherwise use RRF
        ranker = WeightedRanker(*weights) if weights else RRFRanker()
        
        results = self.collection.hybrid_search(
            search_reqs,
            ranker,
            limit=limit,
            output_fields=output_fields
        )
        
        end_time = time.time()
        print(self.search_latency_fmt.format(end_time - start_time))
        
        return self._format_results(results[0])

    def text_enhanced_search(
        self,
        query_text: str,
        technical_terms: Optional[List[str]] = None,
        limit: int = 10,
        output_fields: List[str] = ["arxiv_url_link", "abstract", "technical_terms", "key_points"]
    ) -> List[Dict]:
        """
        Perform text-enhanced search combining BM25 with filters for technical terms.
        
        Args:
            query_text: The search query text
            technical_terms: Optional list of technical terms to search for
            limit: Number of results to return
            output_fields: Fields to include in results
        """
        start_time = time.time()
        
        # Construct expression for technical terms if provided
        expr = None
        if technical_terms:
            terms_conditions = [f'technical_terms like "{term}"' for term in technical_terms]
            expr = " or ".join(terms_conditions)
        
        search_params = {
            "metric_type": "BM25",
            "params": {}
        }
        
        results = self.collection.search(
            data=[query_text],
            anns_field="text",
            param=search_params,
            expr=expr,
            limit=limit,
            output_fields=output_fields
        )
        
        end_time = time.time()
        print(self.search_latency_fmt.format(end_time - start_time))
        
        return self._format_results(results[0])

    def _format_results(self, results) -> List[Dict]:
        """Format search results into a clean dictionary format."""
        formatted_results = []
        for hit in results:
            result = {
                "score": hit.score,
                **hit.fields
            }
            formatted_results.append(result)
        return formatted_results

def run_search_tests(collection_name: str = "arxiv_documents"):
    """Run comprehensive search tests on the collection."""
    collection = Collection(collection_name)
    tester = MilvusSearchTester(collection)
    
    print("\n=== Testing Basic Vector Search ===")
    # Generate a random query vector for testing
    query_vector = np.random.rand(4096)  # Match your embedding dimension
    results = tester.basic_vector_search(query_vector)
    print(f"Found {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"\nResult {i}:")
        print(f"URL: {result['arxiv_url_link']}")
        print(f"Category: {result['category']}")
        print(f"Year: {result['year']}")
        print(f"Score: {result['score']}")

    print("\n=== Testing Hybrid Category Search ===")
    results = tester.hybrid_category_search(
        query_vector,
        category="cs.AI",
        year_range=(2020, 2024)
    )
    print(f"Found {len(results)} results in cs.AI from 2020-2024")
    for i, result in enumerate(results[:3], 1):
        print(f"\nResult {i}:")
        print(f"URL: {result['arxiv_url_link']}")
        print(f"Year: {result['year']}")
        print(f"Score: {result['score']}")

    print("\n=== Testing Text-Enhanced Search ===")
    results = tester.text_enhanced_search(
        query_text="transformer architecture improvements",
        technical_terms=["attention mechanism", "self-attention", "transformer"]
    )
    print(f"Found {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"\nResult {i}:")
        print(f"URL: {result['arxiv_url_link']}")
        print(f"Technical Terms: {result['technical_terms']}")
        print(f"Score: {result['score']}")

if __name__ == "__main__":
    run_search_tests()