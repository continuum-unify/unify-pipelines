from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.layout import Layout
import json
import logging

from src.models.model import get_model_interface
from src.models.embedding import embedding_client
from src.connection.milvus_client import milvus_client
from src.logs.logger import setup_logger

logger = setup_logger(__name__)
console = Console()

@dataclass
class Source:
    """Structured source information."""
    doc_id: int
    file_name: str
    title: str
    abstract: str
    text: str
    summary: str
    key_points: str
    technical_terms: str
    relationships: str
    timestamp: int
    score: float

    def to_dict(self) -> dict:
        """Convert source to dictionary for logging."""
        return {
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'title': self.title,
            'score': self.score,
            'timestamp': self.timestamp
        }

@dataclass
class RAGResult:
    """Enhanced RAG result with detailed information."""
    question: str
    answer: str
    sources: List[Source]
    search_time: float
    llm_time: float
    total_time: float
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert result to dictionary for logging."""
        return {
            'question': self.question,
            'answer': self.answer,
            'sources': [s.to_dict() for s in self.sources],
            'metrics': {
                'search_time': self.search_time,
                'llm_time': self.llm_time,
                'total_time': self.total_time
            },
            'metadata': self.metadata
        }

class RAGEngine:
    """Enhanced RAG Engine with CAGRA optimization and comprehensive source handling."""
    
    def __init__(self):
        """Initialize RAG Engine with optimized configurations."""
        self.model = get_model_interface()
        self.milvus_client = milvus_client
        self.embedding_client = embedding_client
        
        # Optimized CAGRA search configuration
        self.SEARCH_CONFIG = {
            "metric_type": "L2",
            "params": {
                "search_width": 128,  # Aligned with CAGRA config
                "ef_search": 100,     # For better recall
                "nprobe": 16          # Optimized for speed/accuracy trade-off
            }
        }
        
        # Enhanced system prompt
        self.SYSTEM_PROMPT = """You are a knowledgeable AI research assistant specializing in academic papers.
        When answering questions:
        1. Base your answers strictly on the provided context
        2. Cite papers by their full titles when referencing them
        3. Include relevant information from abstracts to provide better context
        4. If the context doesn't contain enough information, clearly state this
        5. Structure your answers with clear sections when appropriate
        6. When citing multiple sources, explain how they relate or differ
        7. Use technical terms from the provided context when appropriate
        8. Highlight key relationships between concepts
        """

    def search_documents(self, query: str, top_k: int = 3) -> Optional[List[Source]]:
        """Perform optimized vector search with CAGRA parameters."""
        try:
            # Generate query embedding
            embedding = self.embedding_client.get_embedding(query, input_type="query")
            logger.debug(f"Generated embedding for query: {query[:50]}...")

            # Connect to Milvus
            if not self.milvus_client.connect():
                raise ConnectionError("Failed to connect to Milvus")

            # Perform search
            collection = self.milvus_client.get_collection()
            collection.load()
            
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=self.SEARCH_CONFIG,
                limit=top_k,
                output_fields=[
                    "doc_id", "source_file", "text", "summary", "key_points",
                    "technical_terms", "abstract", "timestamp", "relationships"
                ]
            )

            if not results or not results[0]:
                logger.warning("No results found for query")
                return []

            # Process results and remove duplicates
            seen_files = set()
            sources = []
            
            for hit in results[0]:
                file_name = getattr(hit, 'source_file', 'Unknown')
                
                # Skip duplicates based on filename
                if file_name in seen_files:
                    continue
                    
                seen_files.add(file_name)
                
                source = Source(
                    doc_id=getattr(hit, 'doc_id', -1),
                    file_name=file_name,
                    title=self.extract_paper_title(file_name),
                    abstract=getattr(hit, 'abstract', 'No abstract available'),
                    text=getattr(hit, 'text', ''),
                    summary=getattr(hit, 'summary', ''),
                    key_points=getattr(hit, 'key_points', ''),
                    technical_terms=getattr(hit, 'technical_terms', ''),
                    relationships=getattr(hit, 'relationships', ''),
                    timestamp=getattr(hit, 'timestamp', 0),
                    score=float(hit.distance)
                )
                sources.append(source)

            return sources

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return None
        finally:
            if self.milvus_client:
                collection.release()
                self.milvus_client.disconnect()

    def format_context(self, sources: List[Source]) -> str:
        """Format sources into comprehensive context with structured sections."""
        context_parts = []
        for i, source in enumerate(sources, 1):
            # Format technical terms as bullet points
            tech_terms = self.format_list_items(source.technical_terms)
            relationships = self.format_list_items(source.relationships, separator='.')
            key_points = self.format_list_items(source.key_points, separator='\n')
            
            context_part = (
                f"Source {i}: {source.title}\n\n"
                f"Abstract:\n{source.abstract}\n\n"
                f"Summary:\n{source.summary}\n\n"
                f"Key Points:\n{key_points}\n\n"
                f"Technical Terms:\n{tech_terms}\n\n"
                f"Relationships:\n{relationships}\n\n"
                f"Relevant Excerpt:\n{source.text[:2000]}\n\n"
                f"Metadata:\n"
                f"- Document ID: {source.doc_id}\n"
                f"- Relevance Score: {source.score:.4f}\n"
                f"- Processing Date: {datetime.fromtimestamp(source.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            context_parts.append(context_part)
        
        return "\n\n" + "\n\n".join(context_parts)

    def format_list_items(self, text: str, separator: str = ',') -> str:
        """Format text into bullet points."""
        if not text:
            return "None provided"
        items = [item.strip() for item in text.split(separator) if item.strip()]
        return '\n'.join(f"• {item}" for item in items)

    def display_result(self, result: RAGResult):
        """Enhanced display with comprehensive source information."""
        layout = Layout()
        layout.split_column(
            Layout(name="question"),
            Layout(name="answer"),
            Layout(name="detailed_sources"),
            Layout(name="metrics")
        )

        # Question Panel
        question_panel = Panel(
            f"[bold]{result.question}[/bold]",
            title="Question",
            border_style="cyan"
        )

        # Answer Panel
        answer_panel = Panel(
            Markdown(result.answer),
            title="Answer",
            border_style="green"
        )

        # Detailed Sources
        sources_layout = Layout()
        for idx, source in enumerate(result.sources, 1):
            source_table = Table(
                title=f"Source {idx}: {source.title}",
                show_header=True,
                header_style="bold magenta",
                title_style="bold blue"
            )
            
            source_table.add_column("Field", style="cyan", width=20)
            source_table.add_column("Content", style="white", width=60)

            # Add fields if they have content
            fields = [
                ("Abstract", self.truncate_text(source.abstract)),
                ("Summary", self.truncate_text(source.summary)),
                ("Key Points", self.format_list_items(source.key_points)),
                ("Technical Terms", self.format_list_items(source.technical_terms)),
                ("Relationships", self.format_list_items(source.relationships)),
                ("Relevance Score", f"{source.score:.4f}"),
                ("Document ID", str(source.doc_id)),
                ("Processing Date", 
                 datetime.fromtimestamp(source.timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            ]

            for field, content in fields:
                if content and content.strip() and content != "None provided":
                    source_table.add_row(field, str(content))

            sources_layout.add_split(Layout(source_table))

        sources_panel = Panel(
            sources_layout,
            title="Source Details",
            border_style="blue"
        )

        # Metrics Panel
        metrics_table = Table(show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        
        metrics = [
            ("Search Time", f"{result.search_time:.2f}s"),
            ("LLM Time", f"{result.llm_time:.2f}s"),
            ("Total Time", f"{result.total_time:.2f}s"),
            ("Sources Found", str(len(result.sources))),
            ("Average Score", f"{result.metadata['avg_score']:.4f}"),
            ("Best Score", f"{result.metadata['best_score']:.4f}")
        ]
        
        for metric, value in metrics:
            metrics_table.add_row(metric, value)
        
        metrics_panel = Panel(
            metrics_table,
            title="Performance Metrics",
            border_style="yellow"
        )

        # Update layout
        layout["question"].update(question_panel)
        layout["answer"].update(answer_panel)
        layout["detailed_sources"].update(sources_panel)
        layout["metrics"].update(metrics_panel)
        
        console.print(layout)

    @staticmethod
    def truncate_text(text: str, max_length: int = 200) -> str:
        """Truncate text with ellipsis."""
        if not text:
            return "None provided"
        return f"{text[:max_length]}..." if len(text) > max_length else text

    def extract_paper_title(self, filename: str) -> str:
        """Extract and format paper title from filename."""
        clean_name = filename.replace('_embedded.json', '')
        year, title = None, clean_name
        
        if len(clean_name) > 4 and clean_name[:4].isdigit():
            year, title = clean_name[:4], clean_name[5:]
            
        formatted_title = title.replace('-', ' ').strip()
        return f"{formatted_title} ({year})" if year else formatted_title

    def answer_question(self, question: str, top_k: int = 3) -> RAGResult:
        """Process question and generate comprehensive answer."""
        start_time = time.time()
        metadata = {"query_timestamp": start_time}
        
        # Search phase
        console.print("\n[cyan]🔎 Searching academic papers...[/cyan]")
        search_start = time.time()
        sources = self.search_documents(question, top_k)
        search_time = time.time() - search_start
        
        if not sources:
            return RAGResult(
                question=question,
                answer="No relevant papers found to answer your question.",
                sources=[],
                search_time=search_time,
                llm_time=0,
                total_time=time.time() - start_time,
                metadata=metadata
            )

        # Format context and update metadata
        context = self.format_context(sources)
        metadata.update({
            "num_sources": len(sources),
            "avg_score": sum(s.score for s in sources) / len(sources),
            "best_score": min(s.score for s in sources)
        })

        # Log context for debugging
        logger.debug(f"Generated context length: {len(context)}")
        
        # Generate answer
        console.print("[cyan]🤔 Analyzing papers and generating answer...[/cyan]")
        llm_start = time.time()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Here are relevant excerpts from academic papers:\n"
                                      f"==================\n"
                                      f"{context}\n"
                                      f"==================\n\n"
                                      f"Based on these papers, please answer this question: {question}"}
        ]
        
        try:
            response = self.model.interact_with_model(
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            llm_time = time.time() - llm_start
            
            if not response or 'choices' not in response:
                raise ValueError("Invalid response from language model")
                
            answer = response['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = "I encountered an error while analyzing the papers."
            llm_time = time.time() - llm_start

        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            search_time=search_time,
            llm_time=llm_time,
            total_time=time.time() - start_time,
            metadata=metadata
        )

def main():
    """Main execution function with enhanced error handling and user interaction."""
    rag_engine = RAGEngine()
    
    console.print("[bold cyan]✨ Welcome to the Academic Paper RAG System![/bold cyan]")
    console.print("[cyan]Ask questions about academic papers in the collection (type 'quit' to exit)[/cyan]")
    console.print("\n[yellow]💡 Tip: Be specific in your questions for better results![/yellow]")
    
    while True:
        try:
            question = console.input("\n[bold cyan]Your question:[/bold cyan] ").strip()
            
            if question.lower() == 'quit':
                console.print("[cyan]👋 Thank you for using the Academic Paper RAG System![/cyan]")
                break
            
            if not question:
                console.print("[yellow]⚠️ Please enter a question.[/yellow]")
                continue
            
            result = rag_engine.answer_question(question)
            rag_engine.display_result(result)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            break
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            console.print(f"[red]❌ An error occurred: {str(e)}[/red]")

if __name__ == "__main__":
    main()