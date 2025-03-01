from src.rag import create_academic_rag
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import argparse
import sys
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import threading
from datetime import datetime
from functools import lru_cache
from collections import OrderedDict

# Initialize console for rich formatting
console = Console()

@dataclass
class CachedContext:
    """Container for cached research context."""
    papers: Dict[str, Dict]
    knowledge_graph: Dict
    temporal_data: Dict
    last_accessed: datetime
    metadata: Dict

class LRUCache:
    """Thread-safe LRU Cache implementation."""
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Dict]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
            
    def put(self, key: str, value: Dict):
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
                
    def __len__(self) -> int:
        """Get cache size."""
        with self.lock:
            return len(self.cache)

class CachedResearchAssistant:
    def __init__(self, engine, cache_size: int = 1000):
        self.engine = engine
        self.response_cache = LRUCache(cache_size)
        self.context_cache = LRUCache(cache_size)
        
        # Statistics for cache performance
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0
        }
        
    def _compute_cache_key(self, question: str) -> str:
        """Generate a deterministic cache key."""
        # Normalize question to improve cache hits
        normalized = " ".join(question.lower().split())
        return f"q:{normalized}"
        
    def _compute_context_key(self, doc_ids: List[str]) -> str:
        """Generate cache key for document context."""
        # Sort doc_ids for consistent keys
        sorted_ids = sorted(doc_ids)
        return f"ctx:{'_'.join(sorted_ids)}"

    def _cache_response(self, question: str, response):
        """Cache a response with its context."""
        cache_key = self._compute_cache_key(question)
        self.response_cache.put(cache_key, {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "sources": [s.doc_id for s in response.sources],
                "cached_at": datetime.now().isoformat()
            }
        })

    def _cache_context(self, doc_ids: List[str], sources: List[Dict]):
        """Cache research context for documents."""
        context_key = self._compute_context_key(doc_ids)
        
        # Organize context data
        context = CachedContext(
            papers={},
            knowledge_graph=self._build_knowledge_graph(sources),
            temporal_data=self._organize_temporal_data(sources),
            last_accessed=datetime.now(),
            metadata={
                "doc_count": len(doc_ids),
                "cached_at": datetime.now().isoformat()
            }
        )
        
        # Process each source
        for source in sources:
            context.papers[source.doc_id] = {
                "title": source.title,
                "abstract": source.abstract,
                "year": source.year,
                "category": source.category,
                "url": source.arxiv_url_link,
                "technical_terms": source.technical_terms if hasattr(source, 'technical_terms') else [],
                "key_points": source.key_points if hasattr(source, 'key_points') else []
            }
            
        self.context_cache.put(context_key, context)

    def answer_question(self, question: str) -> Dict:
        """Enhanced question answering with caching."""
        self.stats["total_queries"] += 1
        
        # Try to get cached response
        cached_response = self.response_cache.get(self._compute_cache_key(question))
        if cached_response:
            self.stats["cache_hits"] += 1
            console.print("[green]Retrieved from cache[/green]")
            return cached_response["response"]

        self.stats["cache_misses"] += 1
        console.print("[cyan]Searching academic papers...[/cyan]")
        
        # Get new response from engine
        response = self.engine.answer_question(question)
        
        # Cache the response and its context
        self._cache_response(question, response)
        self._cache_context(
            [s.doc_id for s in response.sources],
            response.sources
        )
        
        return response

    def _build_knowledge_graph(self, sources: List[Dict]) -> Dict:
        """Build knowledge graph from sources."""
        graph = {
            "nodes": [],
            "edges": [],
            "clusters": {}
        }
        
        for source in sources:
            # Add paper node
            graph["nodes"].append({
                "id": source.doc_id,
                "type": "paper",
                "title": source.title,
                "year": source.year
            })
            
            # Process technical terms if available
            if hasattr(source, 'technical_terms'):
                terms = source.technical_terms
                for term in terms:
                    # Add term node if not exists
                    if not any(n["id"] == term for n in graph["nodes"]):
                        graph["nodes"].append({
                            "id": term,
                            "type": "term"
                        })
                    
                    # Add edge
                    graph["edges"].append({
                        "source": source.doc_id,
                        "target": term,
                        "type": "contains"
                    })
        
        return graph

    def _organize_temporal_data(self, sources: List[Dict]) -> Dict:
        """Organize sources by year."""
        temporal = {}
        
        for source in sources:
            year = source.year
            if year not in temporal:
                temporal[year] = []
                
            temporal[year].append({
                "id": source.doc_id,
                "title": source.title,
                "category": source.category,
                "key_points": source.key_points if hasattr(source, 'key_points') else []
            })
            
        return temporal

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        hit_rate = (
            self.stats["cache_hits"] / 
            max(1, self.stats["total_queries"])
        ) * 100
        
        return {
            "hit_rate": f"{hit_rate:.2f}%",
            "hits": self.stats["cache_hits"],
            "misses": self.stats["cache_misses"],
            "total_queries": self.stats["total_queries"],
            "response_cache_size": len(self.response_cache),
            "context_cache_size": len(self.context_cache)
        }

# Rest of the main script remains the same...

def main():
    # Initialize the RAG engine with caching
    console.print("[cyan]Initializing Research Assistant...[/cyan]")
    engine = create_academic_rag()
    assistant = CachedResearchAssistant(engine)
    console.print("[green]Assistant Ready![/green]\n")

    # Command help
    HELP_TEXT = """
    [bold]/help[/bold] - Show this help message
    [bold]/quit[/bold] - Exit the assistant
    [bold]/export [filename][/bold] - Export responses to a file
    [bold]/sources[/bold] - View sources from the last response
    [bold]/stats[/bold] - View cache statistics
    """

    console.print(Panel(HELP_TEXT, title="Commands", border_style="magenta"))
    
    responses = []
    last_response = None

    while True:
        try:
            question = console.input("[bold cyan]Research Question[/bold cyan]: ")
            
            if question.startswith("/help"):
                console.print(Panel(HELP_TEXT, title="Commands", border_style="magenta"))
                continue
            elif question.startswith("/quit"):
                console.print("[green]Goodbye![/green]")
                break
            elif question.startswith("/export"):
                handle_export(question, responses)
                continue
            elif question.startswith("/sources"):
                handle_sources(last_response)
                continue
            elif question.startswith("/stats"):
                stats = assistant.get_cache_stats()
                display_cache_stats(stats)
                continue

            console.print("[cyan]Processing your question...[/cyan]")
            response = assistant.answer_question(question)
            display_response(response)
            responses.append(response)
            last_response = response

        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]")

if __name__ == "__main__":
    main()