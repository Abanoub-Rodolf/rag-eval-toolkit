import json
import os
import sys

import click
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .backends import AnthropicBackend, GeminiBackend, LiteLLMBackend, OllamaBackend, OpenAIBackend
from .evaluator import RAGEvaluator
from .metrics import (
    AnswerCompletenessMetric,
    AnswerRelevancyMetric,
    ChunkAttributionMetric,
    CoherenceMetric,
    ConcisenessMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextUtilizationMetric,
    FaithfulnessMetric,
    GroundednessMetric,
    HallucinationMetric,
    SemanticSimilarityMetric,
    ToxicityMetric,
)
from .reliability import measure_reliability
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


class _SetupError(Exception):
    """Raised by _setup_run to signal a message was already printed and the
    command should exit 1."""


def _setup_run(config, dataset, backend, model, cache_path, base_url=None):
    """Shared config/dataset/backend/metric resolution for `run` and
    `reliability`. Prints its own error and raises _SetupError on failure so
    callers can exit(1) without duplicating the error-handling boilerplate.

    Returns:
        (evaluator, data, backend_name, cfg)
    """
    cfg: dict = {}
    if config:
        try:
            with open(config) as f:
                cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            console.print(f"[red]error:[/red] could not parse YAML config: {exc}")
            raise _SetupError from exc

    dataset_path = dataset or cfg.get("dataset")
    backend_name = backend or cfg.get("backend", "openai")
    model_name = model or cfg.get("model")
    effective_cache = cache_path or cfg.get("cache_path")
    effective_base_url = base_url or cfg.get("base_url")
    metrics_list = cfg.get("metrics", ["faithfulness", "answer_relevancy"])

    if not dataset_path:
        console.print("[red]error:[/red] dataset path required (--dataset or config)")
        raise _SetupError

    if backend_name not in BACKEND_MAP:
        console.print(
            f"[red]error:[/red] unknown backend '{backend_name}' "
            f"(valid: {', '.join(sorted(BACKEND_MAP))})"
        )
        raise _SetupError
    backend_cls = BACKEND_MAP[backend_name]
    backend_kwargs = {}
    if model_name:
        backend_kwargs["model"] = model_name
    if effective_base_url:
        # self-hosted backends (Ollama) take the server address here; SDK
        # backends reject it via their own __init__ below with a clear error
        backend_kwargs["base_url"] = effective_base_url

    try:
        engine = backend_cls(**backend_kwargs)
    except Exception as exc:
        console.print(f"[red]error:[/red] could not init {backend_name} backend: {exc}")
        raise _SetupError from exc

    evaluator = RAGEvaluator(engine, cache_path=effective_cache)

    unknown = [m for m in metrics_list if m not in METRIC_MAP]
    for m in unknown:
        console.print(f"[yellow]warning:[/yellow] unknown metric '{m}' skipped")
    for m_name in metrics_list:
        if m_name in METRIC_MAP:
            evaluator.add_metric(METRIC_MAP[m_name]())

    if not evaluator.metrics:
        console.print("[red]error:[/red] no valid metrics configured")
        raise _SetupError

    try:
        data = load_dataset(dataset_path)
    except Exception as exc:
        console.print(f"[red]error:[/red] could not load dataset: {exc}")
        raise _SetupError from exc

    return evaluator, data, backend_name, cfg


def _warn_on_errors(results: dict) -> None:
    for name, count in results.get("errors", {}).items():
        if count:
            console.print(
                f"[yellow]warning:[/yellow] {name}: {count} sample(s) failed to "
                "score (backend or parse error) and were excluded from the average"
            )


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
@click.option("--base-url", default=None, help="Server URL for self-hosted backends (Ollama); also settable as base_url in config")
@click.option("--report", is_flag=True, help="Generate HTML report alongside JSON output")
def run(config, dataset, backend, model, output, cache_path, base_url, report):
    """Run evaluation based on config or CLI flags."""
    try:
        evaluator, data, backend_name, cfg = _setup_run(config, dataset, backend, model, cache_path, base_url=base_url)
    except _SetupError:
        sys.exit(1)

    output_path = output or cfg.get("output", "report.json")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.access(out_dir, os.W_OK):
        console.print(f"[red]error:[/red] output dir not writable: {out_dir}")
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
    _warn_on_errors(results)

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
@click.option("--config", type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--dataset", type=click.Path(exists=True), help="Path to dataset (JSON, JSONL, CSV)")
@click.option("--backend", type=click.Choice(list(BACKEND_MAP.keys())), default=None, help="LLM backend")
@click.option("--model", default=None, help="Model name for the backend")
@click.option("--base-url", default=None, help="Server URL for self-hosted backends (Ollama); also settable as base_url in config")
@click.option("--runs", default=3, show_default=True, help="Number of repeated evaluation passes")
def reliability(config, dataset, backend, model, base_url, runs):
    """Measure judge test-retest reliability: run the dataset multiple times
    and report how much scores move between runs. This measures judge
    self-consistency, not judge-human agreement -- see README Limitations."""
    if runs < 2:
        console.print("[red]error:[/red] --runs must be >= 2")
        sys.exit(1)

    try:
        evaluator, data, backend_name, _cfg = _setup_run(config, dataset, backend, model, None, base_url=base_url)
    except _SetupError:
        sys.exit(1)

    console.print(
        f"[bold]Measuring reliability[/bold]: {len(data)} samples, "
        f"{len(evaluator.metrics)} metrics, {runs} runs, backend={backend_name}"
    )

    report_data = measure_reliability(evaluator, data, runs=runs)

    table = Table(title=f"Judge Reliability ({runs} runs, self-consistency only)")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean stdev", justify="right")
    table.add_column("Max stdev", justify="right")
    table.add_column("Mean range", justify="right")
    table.add_column("Samples", justify="right")
    for name, stats in report_data["metrics"].items():
        table.add_row(
            name,
            "-" if stats["mean_stdev"] is None else f"{stats['mean_stdev']:.4f}",
            "-" if stats["max_stdev"] is None else f"{stats['max_stdev']:.4f}",
            "-" if stats["mean_range"] is None else f"{stats['mean_range']:.4f}",
            str(stats["samples_measured"]),
        )
    console.print(table)


@main.command()
def init():
    """Write a starter evaluation config to eval_config.yaml."""
    template = {
        "dataset": "data.jsonl",
        "backend": "openai",
        "model": "gpt-5.1",
        "metrics": ["faithfulness", "answer_relevancy", "hallucination"],
        "output": "results.json",
        "report": True,
    }
    with open("eval_config.yaml", "w") as f:
        yaml.dump(template, f, default_flow_style=False)
    console.print("Created [bold]eval_config.yaml[/bold]")


if __name__ == "__main__":
    main()
