from typing import Optional, List, Dict, Any
from time import time

from rich.console import Console

from src.logs.logger import setup_logger
from src.model import get_model_interface
from .schemas import RAGResponse, SearchMetrics, ModelMetrics, ArxivSource
from .retriever import create_retriever, BaseRetriever
from .formatter import create_formatter, BaseFormatter

logger = setup_logger(__name__)
console = Console()

class RAGEngine:
    """Main RAG engine that orchestrates retrieval, formatting, and generation."""
    
    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        formatter: Optional[BaseFormatter] = None,
        model_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the RAG engine.
        
        Args:
            retriever: Document retriever instance
            formatter: Context formatter instance
            model_client: LLM client instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.retriever = retriever or create_retriever(
            retriever_type=self.config.get('retriever_type', 'milvus')
        )
        self.formatter = formatter or create_formatter(
            formatter_type=self.config.get('formatter_type', 'academic')
        )
        self.model = model_client or get_model_interface()
        
        # Model generation parameters
        self.generation_params = {
            'max_tokens': self.config.get('max_tokens', 2000),
            'temperature': self.config.get('temperature', 0.3),
            'top_p': self.config.get('top_p', 1),
            'frequency_penalty': self.config.get('frequency_penalty', 0),
            'presence_penalty': self.config.get('presence_penalty', 0)
        }
        
        logger.info("RAG Engine initialized successfully")

    def retrieve_sources(self, query: str) -> tuple[List[ArxivSource], SearchMetrics]:
        """Retrieve relevant sources and track metrics.
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (sources, metrics)
        """
        start_time = time()
        
        try:
            # Generate embedding and search
            embedding_start = time()
            sources = self.retriever.retrieve(query)
            embedding_time = time() - embedding_start
            
            search_time = time() - start_time
            
            metrics = SearchMetrics(
                start_time=start_time,
                embedding_time=embedding_time,
                search_time=search_time,
                total_time=time() - start_time,
                num_results=len(sources),
                avg_score=sum(s.score for s in sources) / len(sources) if sources else 0.0
            )
            
            return sources, metrics
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise

    def generate_response(
        self,
        question: str,
        context: str
    ) -> tuple[str, ModelMetrics, Optional[Dict[str, Any]]]:
        """Generate response using the model.
        
        Args:
            question: Original question
            context: Formatted context
            
        Returns:
            Tuple of (answer, metrics, raw_response)
        """
        start_time = time()
        
        try:
            # Prepare context and generate prompt
            context_start = time()
            prompt = self.formatter.generate_prompt(question, context)
            context_time = time() - context_start
            
            # Generate response
            inference_start = time()
            response = self.model.interact_with_model(
                messages=prompt,
                **self.generation_params
            )
            inference_time = time() - inference_start
            
            if not response or 'choices' not in response:
                raise ValueError("Invalid model response")
            
            answer = response['choices'][0]['message']['content']
            
            metrics = ModelMetrics(
                start_time=start_time,
                context_time=context_time,
                inference_time=inference_time,
                total_time=time() - start_time,
                prompt_tokens=response.get('usage', {}).get('prompt_tokens'),
                completion_tokens=response.get('usage', {}).get('completion_tokens')
            )
            
            return answer, metrics, response
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise

    def answer_question(self, question: str) -> RAGResponse:
        """Process a question through the full RAG pipeline.
        
        Args:
            question: Research question
            
        Returns:
            Structured RAG response
        """
        try:
            # Search phase
            console.print("\n[cyan]Searching academic papers...[/cyan]")
            sources, search_metrics = self.retrieve_sources(question)
            
            if not sources:
                return RAGResponse(
                    question=question,
                    answer="No relevant academic papers found for your query.",
                    sources=[],
                    search_metrics=search_metrics,
                    model_metrics=ModelMetrics(
                        start_time=time(),
                        context_time=0,
                        inference_time=0,
                        total_time=0
                    )
                )

            # Format context
            context = self.formatter.format_context(sources)
            
            # Generate response
            console.print("[cyan]Analyzing papers and generating response...[/cyan]")
            answer, model_metrics, _ = self.generate_response(question, context)
            
            return RAGResponse(
                question=question,
                answer=answer,
                sources=sources,
                search_metrics=search_metrics,
                model_metrics=model_metrics
            )
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return RAGResponse(
                question=question,
                answer="An error occurred while processing your question.",
                sources=[],
                search_metrics=SearchMetrics(
                    start_time=time(),
                    embedding_time=0,
                    search_time=0,
                    total_time=0,
                    num_results=0,
                    avg_score=0.0
                ),
                model_metrics=ModelMetrics(
                    start_time=time(),
                    context_time=0,
                    inference_time=0,
                    total_time=0
                ),
                error=str(e)
            )

    def batch_process(
        self,
        questions: List[str],
        show_progress: bool = True
    ) -> List[RAGResponse]:
        """Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            show_progress: Whether to show progress bar
            
        Returns:
            List of RAG responses
        """
        responses = []
        total = len(questions)
        
        with console.status("[bold green]Processing questions...") as status:
            for i, question in enumerate(questions, 1):
                try:
                    response = self.answer_question(question)
                    responses.append(response)
                    
                    if show_progress:
                        status.update(f"Processed {i}/{total} questions")
                        
                except Exception as e:
                    logger.error(f"Error processing question {i}: {str(e)}")
                    responses.append(
                        RAGResponse(
                            question=question,
                            answer=f"Error: {str(e)}",
                            sources=[],
                            search_metrics=SearchMetrics(
                                start_time=time(),
                                embedding_time=0,
                                search_time=0,
                                total_time=0,
                                num_results=0,
                                avg_score=0.0
                            ),
                            model_metrics=ModelMetrics(
                                start_time=time(),
                                context_time=0,
                                inference_time=0,
                                total_time=0
                            ),
                            error=str(e)
                        )
                    )
        
        return responses

# Factory function for creating RAG engines
def create_rag_engine(
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RAGEngine:
    """Create a configured RAG engine instance."""
    try:
        # Combine config and kwargs
        full_config = {**(config or {}), **kwargs}
        
        # Create components
        retriever = create_retriever(
            retriever_type=full_config.get('retriever_type', 'milvus'),
            search_params=full_config.get('search_params')
        )
        
        formatter = create_formatter(
            formatter_type=full_config.get('formatter_type', 'academic'),
            **full_config.get('formatter_params', {})
        )
        
        # Create engine
        engine = RAGEngine(
            retriever=retriever,
            formatter=formatter,
            config=full_config
        )
        
        logger.info("Successfully created RAG engine")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create RAG engine: {str(e)}")
        raise