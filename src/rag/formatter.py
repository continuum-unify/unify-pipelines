from typing import List, Dict, Any
from abc import ABC, abstractmethod
import textwrap

from src.logs.logger import setup_logger
from .schemas import ArxivSource

logger = setup_logger(__name__)

class BaseFormatter(ABC):
    """Abstract base class for formatting contexts and prompts."""
    
    @abstractmethod
    def format_context(self, sources: List[ArxivSource]) -> str:
        """Format source documents into context string."""
        pass
    
    @abstractmethod
    def generate_prompt(self, question: str, context: str) -> List[Dict[str, str]]:
        """Generate formatted prompt for the model."""
        pass

class AcademicFormatter(BaseFormatter):
    """Formatter specialized for academic paper context and prompts."""
    
    def __init__(self, max_context_length: int = 14000):
        """Initialize the academic formatter.
        
        Args:
            max_context_length: Maximum length for formatted context
        """
        self.max_context_length = max_context_length
        
        self.SYSTEM_PROMPT = textwrap.dedent("""
        You are a specialized AI assistant for academic research papers. 
        When answering questions:
        1. Always reference papers using their numbered format: Paper [X] (YEAR, CATEGORY).
        2. Include the title and URL of each paper when citing.
        3. Begin with a high-level overview of how the papers relate to the question.
        4. For each relevant paper:
           - State its main findings and contributions.
           - Explain technical concepts and methodologies.
           - Highlight unique or significant results.
        5. Compare and contrast findings across papers.
        6. Use specific quotes or data points, citing the source paper.
        7. Maintain academic precision and technical accuracy.
        8. End with a synthesis of key insights across all papers.
        """).strip()

    def format_source(self, index: int, source: ArxivSource) -> str:
        """Format a single source document.
        
        Args:
            index: Source index number
            source: ArxivSource object
            
        Returns:
            Formatted string representation of the source
        """
        return textwrap.dedent(f"""
        [Paper {index}] - {source.year}, {source.category}
        Title: {source.title or "No Title Provided"}
        URL: {source.arxiv_url_link or "No URL"}

        Abstract:
        {source.abstract or "Abstract not available."}

        Key Points:
        {source.key_points or "No key points provided."}

        Technical Terms:
        {source.technical_terms or "No technical terms provided."}

        Identified Relationships:
        {source.relationships or "No relationships identified."}

        Summary:
        {source.summary or "Summary not available."}

        Relevant Excerpt:
        {textwrap.shorten(source.text, width=1500, placeholder="...")}

        -----------------------------------
        """).strip()

    def format_context(self, sources: List[ArxivSource]) -> str:
        """Format all sources into a context string.
        
        Args:
            sources: List of ArxivSource objects
            
        Returns:
            Formatted context string
        """
        if not sources:
            return "No relevant academic papers found."

        # Format each source
        context_parts = []
        total_length = 0
        
        for i, source in enumerate(sources, 1):
            formatted_source = self.format_source(i, source)
            source_length = len(formatted_source)
            
            # Check if adding this source would exceed max length
            if total_length + source_length > self.max_context_length:
                logger.warning(f"Truncating context after {i-1} sources due to length limit")
                break
                
            context_parts.append(formatted_source)
            total_length += source_length
        
        logger.debug(f"Formatted context with {len(context_parts)} sources, {total_length} chars")
        bibliography = self.append_bibliography(sources[:len(context_parts)])
        return "\n\n".join(context_parts) + "\n\n" + bibliography

    def append_bibliography(self, sources: List[ArxivSource]) -> str:
        """Create a bibliography from the list of sources."""
        bibliography = "\n".join(
            f"- Paper [{i}]: \"{source.title or 'No Title Provided'}\" ({source.year}, {source.category}). URL: {source.arxiv_url_link or 'No URL'}"
            for i, source in enumerate(sources, 1)
        )
        return f"Bibliography:\n{bibliography}"

    def generate_prompt(self, question: str, context: str) -> List[Dict[str, str]]:
        """Generate a structured prompt for the model.
        
        Args:
            question: Research question
            context: Formatted context string
            
        Returns:
            List of message dictionaries for the model
        """
        # Create the user prompt
        user_prompt = textwrap.dedent(f"""
        Research Question: {question}

        Relevant Research Papers:
        ====================
        {context}
        ====================

        Please provide a comprehensive analysis that:
        1. Introduces each relevant paper and its key contributions.
        2. Uses specific citations when discussing findings.
        3. Explains technical concepts in detail.
        4. Compares methodologies and results across papers.
        5. Synthesizes the current state of research on this topic.
        """).strip()

        # Return formatted messages
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

class CustomFormatter(BaseFormatter):
    """Configurable formatter with custom prompts and formatting."""
    
    def __init__(
        self,
        system_prompt: str,
        context_template: str,
        source_template: str,
        max_context_length: int = 14000
    ):
        """Initialize with custom templates.
        
        Args:
            system_prompt: System prompt template
            context_template: Overall context template
            source_template: Individual source template
            max_context_length: Maximum context length
        """
        self.system_prompt = system_prompt
        self.context_template = context_template
        self.source_template = source_template
        self.max_context_length = max_context_length

    def format_context(self, sources: List[ArxivSource]) -> str:
        """Format sources using custom template."""
        formatted_sources = [
            self.source_template.format(
                index=i,
                **source.__dict__
            )
            for i, source in enumerate(sources, 1)
        ]
        
        # Combine sources and apply context template
        return self.context_template.format(
            sources="\n\n".join(formatted_sources)
        )

    def generate_prompt(self, question: str, context: str) -> List[Dict[str, str]]:
        """Generate prompt using custom templates."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{question}\n\n{context}"}
        ]

def create_formatter(
    formatter_type: str = "academic",
    **kwargs
) -> BaseFormatter:
    """Create a formatter instance based on specified type.
    
    Args:
        formatter_type: Type of formatter to create
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured formatter instance
    """
    if formatter_type == "academic":
        return AcademicFormatter(**kwargs)
    elif formatter_type == "custom":
        required_keys = {"system_prompt", "context_template", "source_template"}
        missing_keys = required_keys - set(kwargs.keys())
        if missing_keys:
            raise ValueError(f"Missing required templates: {missing_keys}")
        return CustomFormatter(**kwargs)
    else:
        raise ValueError(f"Unknown formatter type: {formatter_type}")
