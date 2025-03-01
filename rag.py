from src.rag import create_academic_rag
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import argparse
import sys
import json

# Initialize console for rich formatting
console = Console()

# Parse arguments for debug mode
parser = argparse.ArgumentParser(description="Run the RAG research assistant")
parser.add_argument('--debug', action='store_true', help="Enable debug logging")
args = parser.parse_args()

if args.debug:
    import logging
    logging.basicConfig(level=logging.DEBUG)

# Initialize the RAG engine
console.print("[cyan]Initializing Research Assistant...[/cyan]")
engine = create_academic_rag()
console.print("[green]Assistant Ready![/green]\n")

# Command help
HELP_TEXT = """
[bold]/help[/bold] - Show this help message
[bold]/quit[/bold] - Exit the assistant
[bold]/export [filename][/bold] - Export responses to a file
[bold]/sources[/bold] - View sources from the last response
"""

responses = []  # Store responses for export
last_response = None  # Track the last response

def display_response(response):
    global last_response
    last_response = response

    # Display the question
    console.print(Panel(f"[bold cyan]{response.question}[/bold cyan]", title="Question", border_style="cyan"))

    # Display the answer
    console.print(Panel(response.answer, title="Answer", border_style="green"))

    # Display sources
    sources_table = Table(title="Sources", show_header=True, header_style="bold magenta")
    sources_table.add_column("ID", justify="right", style="cyan")
    sources_table.add_column("Year", style="green")
    sources_table.add_column("Category", style="blue")
    sources_table.add_column("URL", style="yellow")
    sources_table.add_column("Score", justify="right", style="red")

    for i, source in enumerate(response.sources, start=1):
        sources_table.add_row(
            str(i),
            str(source.year),
            source.category,
            source.arxiv_url_link,  # Correct field name
            f"{source.score:.4f}"
        )

    console.print(sources_table)

    # Display metrics
    metrics_table = Table(title="Performance Metrics", show_header=False)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="yellow")

    for step, value in vars(response.search_metrics).items():  # Corrected to use vars()
        metrics_table.add_row(step.replace('_', ' ').title(), f"{value:.2f}s")

    for step, value in vars(response.model_metrics).items():  # Corrected to use vars()
        metrics_table.add_row(step.replace('_', ' ').title(), f"{value:.2f}s")

    metrics_table.add_row("Total Time", f"{response.total_time:.2f}s")

    console.print(metrics_table)

def handle_export(command):
    parts = command.split()
    if len(parts) != 2:
        console.print("[red]Usage: /export [filename][/red]")
        return

    filename = parts[1]
    try:
        with open(filename, 'w') as f:
            json.dump([response.to_dict() for response in responses], f, indent=2)
        console.print(f"[green]Responses exported to {filename}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to export responses: {str(e)}[/red]")

def handle_sources():
    if not last_response or not last_response.sources:
        console.print("[red]No sources available from the last response.[/red]")
        return

    for i, source in enumerate(last_response.sources, start=1):
        console.print(Panel(
            f"[bold]Source {i}[/bold]\n\n"
            f"Year: {source.year}\n"
            f"Category: {source.category}\n"
            f"URL: {source.arxiv_url_link}\n\n"  # Correct field name
            f"Abstract:\n{source.abstract[:500]}...",
            title=f"Source {i}", border_style="blue"
        ))

def main():
    console.print(Panel(HELP_TEXT, title="Commands", border_style="magenta"))
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
                handle_export(question)
                continue
            elif question.startswith("/sources"):
                handle_sources()
                continue

            console.print("[cyan]Processing your question...[/cyan]")
            response = engine.answer_question(question)
            display_response(response)
            responses.append(response)

        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]")

if __name__ == "__main__":
    main()
