import click
import json
import os
import yaml
from typing import Optional
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
    ContextUtilizationMetric, GroundednessMetric, SemanticSimilarityMetric
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
    "semantic_similarity": SemanticSimilarityMetric
}

BACKEND_MAP = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "ollama": OllamaBackend,
    "gemini": GeminiBackend,
    "litellm": LiteLLMBackend
}

@click.group()
def main():
    """RAG Evaluation Toolkit v1.0 — CLI"""
    pass

@main.command()
def metrics():
    """List all available evaluation metrics."""
    table = Table(title="Available Metrics")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    
    for name, cls in METRIC_MAP.items():
        doc = cls.__doc__.split('\n')[0] if cls.__doc__ else "No description."
        table.add_row(name, doc)
    
    console.print(table)

@main.command()
@click.option('--config', type=click.Path(exists=True), help='Path to YAML config file')
@click.option('--dataset', type=click.Path(exists=True), help='Path to dataset (JSON, JSONL, CSV)')
@click.option('--backend', type=click.Choice(list(BACKEND_MAP.keys())), default=None, help='LLM backend')
@click.option('--model', default=None, help='Model name for the backend')
@click.option('--output', type=click.Path(), default='report.json', help='Output results path')
@click.option('--report', is_flag=True, help='Generate HTML report')
def run(config, dataset, backend, model, output, report):
    """Run evaluation based on config or CLI flags."""
    cfg = {}
    if config:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
    
    # Override config with CLI flags
    dataset_path = dataset or cfg.get('dataset')
    backend_name = backend or cfg.get('backend', 'openai')
    model_name = model or cfg.get('model')
    output_path = output or cfg.get('output', 'report.json')
    metrics_list = cfg.get('metrics', ['faithfulness', 'answer_relevancy'])

    if not dataset_path:
        console.print("[red]Error:[/red] Dataset path is required (via --dataset or config).")
        return

    # Initialize Backend
    backend_cls = BACKEND_MAP[backend_name]
    backend_args = {}
    if model_name:
        backend_args['model'] = model_name
    
    try:
        engine = backend_cls(**backend_args)
    except Exception as e:
        console.print(f"[red]Error initializing backend {backend_name}:[/red] {e}")
        return

    # Initialize Evaluator
    evaluator = RAGEvaluator(engine, cache_path=cfg.get('cache_path', '.rag_eval_cache.db'))
    
    for m_name in metrics_list:
        if m_name in METRIC_MAP:
            evaluator.add_metric(METRIC_MAP[m_name]())
        else:
            console.print(f"[yellow]Warning:[/yellow] Unknown metric '{m_name}' skipped.")

    # Load Data
    try:
        data = load_dataset(dataset_path)
    except Exception as e:
        console.print(f"[red]Error loading dataset:[/red] {e}")
        return

    console.print(f"🚀 [bold]Starting Evaluation[/bold] ({len(data)} samples, {len(evaluator.metrics)} metrics)")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(data))
        
        # We wrap evaluate in a way that allows progress updates if we had a per-item callback.
        # For now, we'll just run it.
        results = evaluator.evaluate(data)
        progress.update(task, advance=len(data))

    # Display Summary
    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Average Score", justify="right", style="green")
    
    for name, avg in results['averages'].items():
        summary_table.add_row(name, f"{avg:.4f}")
    
    console.print(summary_table)

    # Save Results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"✅ Results saved to [bold]{output_path}[/bold]")

    if report or cfg.get('report'):
        report_path = output_path.replace('.json', '.html')
        evaluator.generate_report(results, output=report_path)
        console.print(f"📊 HTML Report generated at [bold]{report_path}[/bold]")

@main.command()
def init():
    """Initialize a new evaluation config template."""
    config_template = {
        "dataset": "data.jsonl",
        "backend": "openai",
        "model": "gpt-4",
        "metrics": ["faithfulness", "answer_relevancy", "hallucination"],
        "output": "results.json",
        "report": True,
        "cache_path": ".rag_eval_cache.db"
    }
    
    with open("eval_config.yaml", "w") as f:
        yaml.dump(config_template, f, default_flow_style=False)
    
    console.print("✨ Created [bold]eval_config.yaml[/bold] template.")

if __name__ == '__main__':
    main()
