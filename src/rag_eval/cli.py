import click
import json
import os
import sys
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .evaluator import RAGEvaluator
from .backends import OpenAIBackend, AnthropicBackend, OllamaBackend, GeminiBackend, LiteLLMBackend
from .metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric,
    ContextPrecisionMetric, ContextRecallMetric,
    HallucinationMetric, ToxicityMetric, CoherenceMetric, ConcisenessMetric,
    AnswerCompletenessMetric, ChunkAttributionMetric,
    ContextUtilizationMetric, GroundednessMetric, SemanticSimilarityMetric,
)
from .utils import load_dataset

console = Console()

METRIC_MAP = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "context_precision": ContextPrecisionMetric,
    "context_recall": ContextRecallMetric,
    "hallucination": HallucinationMetric,
    "toxicity": ToxicityMetric,
    "coherence": CoherenceMetric,
    "conciseness": ConcisenessMetric,
    "completeness": AnswerCompletenessMetric,
    "chunk_attribution": ChunkAttributionMetric,
    "context_utilization": ContextUtilizationMetric,
    "groundedness": GroundednessMetric,
    "semantic_similarity": SemanticSimilarityMetric,
}

BACKEND_MAP = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "ollama": OllamaBackend,
    "gemini": GeminiBackend,
    "litellm": LiteLLMBackend,
}


@click.group()
def main():
    """RAG Evaluation Toolkit CLI."""
    pass


@main.command()
def metrics():
    """List all available evaluation metrics."""
    table = Table(title="Available Metrics")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    for name, cls in METRIC_MAP.items():
        doc = cls.__doc__.split("\n")[0].strip() if cls.__doc__ else "No description."
        table.add_row(name, doc)
    console.print(table)


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--dataset", type=click.Path(exists=True), help="Path to dataset (JSON, JSONL, CSV)")
@click.option("--backend", type=click.Choice(list(BACKEND_MAP.keys())), default=None, help="LLM backend")
@click.option("--model", default=None, help="Model name for the backend")
@click.option("--output", type=click.Path(), default=None, help="Output results path")
@click.option("--cache-path", default=None, help="SQLite cache path (omit to disable caching)")
@click.option("--report", is_flag=True, help="Generate HTML report alongside JSON output")
def run(config, dataset, backend, model, output, cache_path, report):
    """Run evaluation based on config or CLI flags."""
    cfg: dict = {}
    if config:
        try:
            with open(config) as f:
                cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            console.print(f"[red]error:[/red] could not parse YAML config: {exc}")
            sys.exit(1)

    dataset_path = dataset or cfg.get("dataset")
    backend_name = backend or cfg.get("backend", "openai")
    model_name = model or cfg.get("model")
    output_path = output or cfg.get("output", "report.json")
    effective_cache = cache_path or cfg.get("cache_path")
    metrics_list = cfg.get("metrics", ["faithfulness", "answer_relevancy"])

    if not dataset_path:
        console.print("[red]error:[/red] dataset path required (--dataset or config)")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.access(out_dir, os.W_OK):
        console.print(f"[red]error:[/red] output dir not writable: {out_dir}")
        sys.exit(1)

    backend_cls = BACKEND_MAP[backend_name]
    backend_kwargs = {}
    if model_name:
        backend_kwargs["model"] = model_name

    try:
        engine = backend_cls(**backend_kwargs)
    except Exception as exc:
        console.print(f"[red]error:[/red] could not init {backend_name} backend: {exc}")
        sys.exit(1)

    evaluator = RAGEvaluator(engine, cache_path=effective_cache)

    unknown = [m for m in metrics_list if m not in METRIC_MAP]
    for m in unknown:
        console.print(f"[yellow]warning:[/yellow] unknown metric '{m}' skipped")
    for m_name in metrics_list:
        if m_name in METRIC_MAP:
            evaluator.add_metric(METRIC_MAP[m_name]())

    if not evaluator.metrics:
        console.print("[red]error:[/red] no valid metrics configured")
        sys.exit(1)

    try:
        data = load_dataset(dataset_path)
    except Exception as exc:
        console.print(f"[red]error:[/red] could not load dataset: {exc}")
        sys.exit(1)

    total_steps = len(data) * len(evaluator.metrics)
    console.print(
        f"[bold]Starting evaluation[/bold]: {len(data)} samples, "
        f"{len(evaluator.metrics)} metrics, backend={backend_name}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=total_steps)
        results = evaluator.evaluate(data, on_progress=lambda: progress.advance(task, 1))

    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")
    for name, avg in results["averages"].items():
        table.add_row(name, f"{avg:.4f}")
    console.print(table)

    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    except OSError as exc:
        console.print(f"[red]error:[/red] could not write {output_path}: {exc}")
        sys.exit(1)
    console.print(f"Results saved to [bold]{output_path}[/bold]")

    if report or cfg.get("report"):
        base, _ = os.path.splitext(output_path)
        report_path = base + ".html"
        try:
            evaluator.generate_report(results, output=report_path)
        except OSError as exc:
            console.print(f"[red]error:[/red] could not write report: {exc}")
            sys.exit(1)
        console.print(f"HTML report at [bold]{report_path}[/bold]")


@main.command()
def init():
    """Write a starter evaluation config to eval_config.yaml."""
    template = {
        "dataset": "data.jsonl",
        "backend": "openai",
        "model": "gpt-4o",
        "metrics": ["faithfulness", "answer_relevancy", "hallucination"],
        "output": "results.json",
        "report": True,
    }
    with open("eval_config.yaml", "w") as f:
        yaml.dump(template, f, default_flow_style=False)
    console.print("Created [bold]eval_config.yaml[/bold]")


if __name__ == "__main__":
    main()
