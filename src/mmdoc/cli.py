"""mmdoc — Multi-modal Document Understanding. VL model + structured extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mmdoc.extractor import ask_document, batch_extract, describe_document, extract_structured

app = typer.Typer(help="Multi-modal document understanding with Qwen2-VL", no_args_is_help=True)
console = Console()


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            import contextlib
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8")


@app.command()
def extract(
    file_path: str = typer.Argument(help="Path to document (PDF, PNG, JPG, GIF)"),
    prompt: str = typer.Option("Extract all information from this document page", help="What to extract"),
    output: str | None = typer.Option(None, help="Save JSON result to file"),
    temperature: float = typer.Option(0.0, help="Model temperature"),
) -> None:
    """Extract structured data from a document using the VL model."""
    _utf8_stdout()
    with console.status(f"[cyan]Loading model + processing {Path(file_path).name}..."):
        result = extract_structured(file_path, prompt, temperature=temperature)

    for page in result.pages:
        console.print(f"\n[bold]Page {page['page']}[/bold]")
        console.print(page["content"])

    if output:
        Path(output).write_text(json.dumps(result.pages, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[green]Saved to {output}[/green]")


@app.command()
def describe(
    file_path: str = typer.Argument(help="Path to document (PDF, PNG, JPG, GIF)"),
    temperature: float = typer.Option(0.0, help="Model temperature"),
) -> None:
    """Describe what each page/frame of a document depicts in 1-2 sentences."""
    _utf8_stdout()
    with console.status(f"[cyan]Loading model + processing {Path(file_path).name}..."):
        result = describe_document(file_path, temperature=temperature)
    console.print(result)


@app.command()
def ask(
    file_path: str = typer.Argument(help="Path to document"),
    question: str = typer.Argument(help="Question to ask about the document"),
    temperature: float = typer.Option(0.0, help="Model temperature"),
) -> None:
    """Ask a question about a document."""
    _utf8_stdout()
    with console.status(f"[cyan]Loading model + processing {Path(file_path).name}..."):
        answer = ask_document(file_path, question, temperature=temperature)

    console.print(f"\n[cyan]Q:[/cyan] {question}")
    console.print(answer)


@app.command()
def batch(
    glob_pattern: str = typer.Argument(help="File glob, e.g. 'data/*.pdf' or '*.png'"),
    prompt: str = typer.Option("Extract all information from each document", help="What to extract"),
    output: str = typer.Option("batch_results.json", help="Output JSON file"),
    temperature: float = typer.Option(0.0, help="Model temperature"),
) -> None:
    """Process multiple files matching a glob pattern."""
    _utf8_stdout()
    files = sorted(Path().glob(glob_pattern))
    if not files:
        console.print(f"[red]No files matching: {glob_pattern}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Processing {len(files)} files...[/cyan]")
    results = batch_extract([str(f) for f in files], prompt, temperature=temperature)

    out_data = []
    for r in results:
        out_data.append({"path": r.path, "format": r.format, "pages": r.pages})
        table = Table(title=f"{Path(r.path).name} ({r.format}, {len(r.pages)} {'pages' if r.format == 'pdf' else 'frames'})")
        table.add_column("Page", style="dim")
        table.add_column("Content", style="white", max_width=80)
        for p in r.pages:
            table.add_row(str(p["page"]), p["content"][:120])
        console.print(table)

    Path(output).write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]{len(results)} files → {output}[/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Run the FastAPI server."""
    _utf8_stdout()
    import uvicorn

    uvicorn.run("mmdoc.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
