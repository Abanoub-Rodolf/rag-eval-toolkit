"""HTML report generation for evaluation results."""
import html as _html
import json
from typing import Any, Dict


def generate_html_report(results: Dict[str, Any], output_path: str) -> None:
    """Write a self-contained HTML evaluation report to disk.

    Args:
        results: Output from RAGEvaluator.evaluate() — must have 'averages'
            and 'per_sample' keys.
        output_path: Destination file path for the HTML report.
    """
    averages = results.get("averages", {})
    per_sample = results.get("per_sample", {})

    rows_html = "\n".join(
        f"<tr><td>{name}</td><td>{score:.4f}</td>"
        f"<td>{_score_bar(score)}</td></tr>"
        for name, score in averages.items()
    )

    sample_rows = ""
    if per_sample:
        metrics = list(per_sample.keys())
        n = max(len(v) for v in per_sample.values()) if per_sample else 0
        header = "<tr><th>#</th>" + "".join(f"<th>{m}</th>" for m in metrics) + "</tr>"
        body = ""
        for i in range(n):
            cols = "".join(
                f"<td>{per_sample[m][i]:.4f}</td>" for m in metrics
            )
            body += f"<tr><td>{i + 1}</td>{cols}</tr>"
        sample_rows = f"<h2>Per-Sample Scores</h2><table>{header}{body}</table>"

    raw_json = _html.escape(json.dumps(results, indent=2))
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Evaluation Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; color: #1a1a1a; }}
  h1 {{ border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
  th {{ background: #f7f7f7; font-weight: 600; }}
  .bar {{ background: #e2f0fb; border-radius: 4px; height: 14px; display: inline-block; }}
  pre {{ background: #f7f7f7; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 12px; }}
</style>
</head>
<body>
<h1>RAG Evaluation Report</h1>
<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Score</th><th>Visual</th></tr>
  {rows_html}
</table>
{sample_rows}
<h2>Raw Results</h2>
<pre>{raw_json}</pre>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def _score_bar(score: float) -> str:
    pct = int(score * 100)
    return f'<span class="bar" style="width:{pct}px">&nbsp;</span> {pct}%'


class HTMLReportGenerator:
    """Thin class wrapper around generate_html_report for backwards compat."""

    def generate(self, results: Dict[str, Any], output_path: str) -> None:
        generate_html_report(results, output_path)
