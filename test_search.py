"""
Complete Milvus Search Implementation for arXiv papers with multiple search strategies.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import time
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
import textwrap
from pymilvus import Collection
from src import MilvusClient, milvus_client, EmbeddingClient, embedding_client

console = Console()

class MilvusSearchTester:
    """Handles various search operations on a Milvus collection."""
    
    def __init__(self, collection: Collection):
        self.collection = collection
        
    def vector_search(
        self, 
        query_vector: np.ndarray,
        expr: Optional[str] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Base vector search method with optional filtering."""
        if output_fields is None:
            output_fields = ["arxiv_url_link", "summary", "year", "category"]
            
        start_time = time.time()
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
        
        print(f"Search latency = {time.time() - start_time:.4f}s")
        return [
            {
                "score": hit.distance,
                **hit.fields
            }
            for hit in results[0]
        ]

    def basic_vector_search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform basic vector similarity search."""
        return self.vector_search(
            query_vector=query_vector,
            limit=limit,
            output_fields=output_fields
        )

    def hybrid_category_search(
        self,
        query_vector: np.ndarray,
        category: str,
        year_range: Optional[tuple] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with category and year filtering."""
        expr = f'category == "{category}"'
        if year_range:
            expr += f' and year >= {year_range[0]} and year <= {year_range[1]}'
            
        return self.vector_search(
            query_vector=query_vector,
            expr=expr,
            limit=limit,
            output_fields=output_fields
        )

    def text_enhanced_search(
        self,
        query_vector: np.ndarray,
        technical_terms: Optional[List[str]] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector search with technical terms filtering."""
        if output_fields is None:
            output_fields = ["arxiv_url_link", "summary", "year", "category", "technical_terms"]
            
        expr = None
        if technical_terms:
            terms_conditions = [f'technical_terms like "%{term}%"' for term in technical_terms]
            expr = " or ".join(terms_conditions)
            
        return self.vector_search(
            query_vector=query_vector,
            expr=expr,
            limit=limit,
            output_fields=output_fields
        )

class SearchDemo:
    """Interactive demo for searching arXiv papers using various strategies."""
    
    def __init__(self):
        """Initialize search demo with necessary clients and connections."""
        self.milvus_client = milvus_client
        self.embedding_client = embedding_client
        
        if not self.embedding_client:
            raise RuntimeError("Embedding client initialization failed")
        
        if not self.milvus_client.connect():
            raise ConnectionError("Failed to connect to Milvus")
        
        if not self.milvus_client.create_collection():
            raise ConnectionError("Failed to create/get collection")
        
        self.collection = self.milvus_client.get_collection()
        if not self.collection:
            raise ConnectionError("Failed to get collection")
        
        if not self.milvus_client.load_collection():
            raise ConnectionError("Failed to load collection")
        
        self.search_tester = MilvusSearchTester(self.collection)
        console.print("[green]Successfully connected to Milvus and loaded collection[/green]")

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query text."""
        embedding = self.embedding_client.get_embedding(query, input_type="passage")
        if embedding is None:
            raise ValueError("Failed to generate embedding for query")
        return np.array(embedding)

    def print_results(self, results: List[Dict[str, Any]], title: str):
        """Display search results in a formatted table."""
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(title=title)
        table.add_column("Score", justify="right", style="cyan", width=10)
        table.add_column("Year", justify="right", style="green", width=6)
        table.add_column("Category", style="magenta", width=15)
        table.add_column("Summary", style="white", width=60, overflow="fold")
        table.add_column("URL", style="blue", width=40)

        for result in results:
            summary = textwrap.shorten(result.get('summary', ''), width=200, placeholder="...")
            table.add_row(
                f"{result.get('score', 0.0):.4f}",
                str(result.get('year', 'N/A')),
                result.get('category', 'N/A'),
                summary,
                result.get('arxiv_url_link', 'N/A')
            )

        console.print(table)

    def semantic_search(self):
        """Perform semantic search using text query."""
        try:
            query = Prompt.ask("\n[yellow]Enter your research query")
            top_k = IntPrompt.ask("[yellow]How many results would you like", default=5)
            
            console.print("\n[cyan]Generating embedding for your query...[/cyan]")
            query_vector = self.get_query_embedding(query)
            
            console.print("\n[cyan]Searching for similar papers...[/cyan]")
            results = self.search_tester.basic_vector_search(
                query_vector=query_vector,
                limit=top_k
            )
            
            self.print_results(results, f"Semantic Search Results for: {query}")
        except Exception as e:
            console.print(f"[red]Error during semantic search: {str(e)}[/red]")

    def category_filtered_search(self):
        """Perform search within specific category and year range."""
        try:
            query = Prompt.ask("\n[yellow]Enter your research query")
            category = Prompt.ask("[yellow]Enter category (e.g., cs.AI, cs.CL, physics.comp-ph)")
            start_year = IntPrompt.ask("[yellow]Start year", default=2015)
            end_year = IntPrompt.ask("[yellow]End year", default=2024)
            top_k = IntPrompt.ask("[yellow]How many results would you like", default=5)

            console.print("\n[cyan]Generating embedding for your query...[/cyan]")
            query_vector = self.get_query_embedding(query)

            console.print("\n[cyan]Searching for papers...[/cyan]")
            results = self.search_tester.hybrid_category_search(
                query_vector=query_vector,
                category=category,
                year_range=(start_year, end_year),
                limit=top_k
            )

            self.print_results(
                results, 
                f"Category Search Results\nQuery: {query}\nCategory: {category}\nYears: {start_year}-{end_year}"
            )
        except Exception as e:
            console.print(f"[red]Error during category search: {str(e)}[/red]")

    def technical_term_search(self):
        """Search using technical terms and vector similarity."""
        try:
            query = Prompt.ask("\n[yellow]Enter your research topic")
            console.print("\n[yellow]Enter technical terms (comma-separated) or press Enter to skip")
            terms_input = Prompt.ask("[yellow]Terms")
            terms = [t.strip() for t in terms_input.split(",")] if terms_input else None
            top_k = IntPrompt.ask("[yellow]How many results would you like", default=5)

            console.print("\n[cyan]Generating embedding for your query...[/cyan]")
            query_vector = self.get_query_embedding(query)

            console.print("\n[cyan]Searching for papers...[/cyan]")
            results = self.search_tester.text_enhanced_search(
                query_vector=query_vector,
                technical_terms=terms,
                limit=top_k
            )

            title = f"Technical Search Results\nQuery: {query}"
            if terms:
                title += f"\nTerms: {', '.join(terms)}"
            self.print_results(results, title)
        except Exception as e:
            console.print(f"[red]Error during technical term search: {str(e)}[/red]")

    def cleanup(self):
        """Clean up resources before exiting."""
        try:
            if self.milvus_client:
                self.milvus_client.release_collection()
                self.milvus_client.disconnect()
                console.print("[green]Successfully cleaned up resources[/green]")
        except Exception as e:
            console.print(f"[red]Error during cleanup: {str(e)}[/red]")

    def run_demo(self):
        """Run the interactive search demo."""
        try:
            console.print("\n[bold green]=== Milvus arXiv Search Demo ===[/bold green]")
            
            while True:
                console.print("\n[bold yellow]Choose a search type:[/bold yellow]")
                console.print("1. Semantic Search")
                console.print("2. Category & Year Filtered Search")
                console.print("3. Technical Term Search")
                console.print("4. Exit")
                
                choice = Prompt.ask("\n[yellow]Enter your choice", choices=["1", "2", "3", "4"])
                
                if choice == "1":
                    self.semantic_search()
                elif choice == "2":
                    self.category_filtered_search()
                elif choice == "3":
                    self.technical_term_search()
                else:
                    console.print("\n[bold green]Thank you for using the search demo![/bold green]")
                    break
        finally:
            self.cleanup()

def main():
    """Main entry point for the search demo."""
    try:
        demo = SearchDemo()
        demo.run_demo()
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
