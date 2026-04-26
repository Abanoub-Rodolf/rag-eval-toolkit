# Proof manifest — rag-eval-toolkit audit/v1
**Date:** 2026-04-25
**Status:** verified-done

## Scope (original task)
"discovery first then deep audit + improve everything, premium quality, take days, GitLab origin, audit/v1 branch only, no auto-MR, no publishes, no AI attribution, no breaking changes"

## What was done

### Phase 1 (prior session, 14 commits)
- 4 parallel codebase mapper agents (.planning/codebase/ — gitignored)
- Initial bug sweep: report module rewrite, asyncio deprecation, model defaults, CLI exit codes, score parser refactor, cache thread safety, Gemini SDK migration (google-generativeai → google-genai)
- 2 code-review passes (Sonnet)
- Tests grown from ~26 → 95

### Phase 2 (this session, 5 commits, +14 tests)
- 4 parallel **Opus** audit agents: security, correctness, AI-fingerprint, docs/examples
- Findings synthesized, prioritized, applied:

| Severity | Finding | Fix | File |
|----------|---------|-----|------|
| CRITICAL | XSS in HTML report metric names | `_html.escape(str(name))` everywhere | report/generator.py |
| HIGH | Score bar overflow (>1.0/<0.0) | Clamp `max(0, min(100, ...))` | report/generator.py |
| HIGH | Bad YAML config crashes traceback | try/except + sys.exit(1) | cli.py |
| HIGH | Unwritable output crashes after eval runs | `os.access` pre-check + try/except | cli.py |
| HIGH | on_progress missing on exception | finally block in score_task | evaluator.py |
| HIGH | issubclass(OpenAIBackend, OpenAIBackend) → False | ABCMeta metaclass + (BaseBackend,) bases + self short-circuit | backends/__init__.py |
| HIGH | UTF-8 BOM crashes JSONL/breaks CSV | encoding="utf-8-sig" everywhere | utils/loaders.py |
| HIGH | JSONL doesn't skip comment lines | Filter `#`-prefixed lines | utils/loaders.py |
| HIGH | Retired claude-3-opus in litellm docstring | Replace with current alias | backends/litellm_backend.py |
| HIGH | OpenAI None content crashes parser | Coerce to "" | backends/openai_backend.py |
| HIGH | CI installs ghost pydantic dep | `pip install -e ".[dev]"` | .github/workflows/ci.yml |
| MEDIUM | README claims jinja2 needed | Stdlib only | README.md |
| MEDIUM | Stale GitHub clone URLs | GitLab clone URLs | README.md, docs/getting_started.md |
| MEDIUM | Dead GitHub Actions badge | Removed | README.md |
| LOW | 7 em dashes (banned per rules) | Replaced with periods/colons | cli.py, evaluator.py, faithfulness.py, cache.py |
| LOW | Trivial docstrings | Dropped | semantic_similarity.py, ollama_backend.py |

## Verification

### Tests
```
$ python -m pytest tests/ -q
109 passed in 1.20s
```

### Imports + lazy loading
```
$ python -c "from rag_eval.backends import OpenAIBackend, BaseBackend; print(issubclass(OpenAIBackend, BaseBackend))"
True
$ python -c "import sys; from rag_eval.backends import OpenAIBackend; print('openai' in sys.modules)"
False  # lazy loading confirmed
```

### CLI smoke
```
$ python -m rag_eval.cli --help        # OK, shows init/metrics/run
$ python -m rag_eval.cli metrics       # OK, lists all 13 metrics
```

### End-to-end (stub backend, /tmp/rag_eval_smoke/smoke.py)
```
on_progress fires: 4 (expected 4)
results.averages: {'faithfulness': 0.85, 'answer_relevancy': 0.85}
report.html size: 1546 bytes
contains <script>: False    # XSS verified blocked
```

### Branch + remote
```
$ git log main..audit/v1 --oneline | wc -l
19   # 19 commits on audit/v1

$ git ls-remote --heads origin
4efb5813...  refs/heads/main
e38a279...   refs/heads/audit/v1   # both synced

$ git remote -v
origin  git@gitlab.com:abanoub.rodolf/rag-eval-toolkit.git (fetch)
origin  git@gitlab.com:abanoub.rodolf/rag-eval-toolkit.git (push)
```

### AI attribution scan (final)
```
$ git log main..audit/v1 --format="%an <%ae>" | sort -u
Abanoub Rodolf Boctor <abanoub.rodolf@gmail.com>

$ git log main..audit/v1 --format="%H%n%B%n---" | grep -iE "co-authored-by|claude|generated with"
CLEAN
```

## What was NOT done (per original task constraints)

- ❌ No merge request opened (audit/v1 → main) — "no auto-MR"
- ❌ No PyPI publish — "no publishes"
- ❌ No history rewrite — "no breaking changes"
- ❌ Did not refactor 12 cloned LLM-judge metric files into a registry — would be a structural change requiring user sign-off (flagged by AI-fingerprint audit but invasive)

## Remaining risk / known limitations

1. **Prompt injection via LLM judge**: All 13 metrics interpolate user-controlled context/answer/question into XML-tagged prompts. XML wrapping is NOT a security boundary — an attacker-controlled context containing `</context>` can manipulate the judge. This is a fundamental limitation of LLM-judge frameworks. Should be documented in README threat model. (HIGH severity in security audit, deferred.)

2. **GitHub badge is dead**: Removed, but the package CI is now untested in cloud (local only). If wanted, add `.gitlab-ci.yml` for GitLab Pipelines. (Out of audit scope.)

3. **All-Cloned Metric Pattern**: 12 of 13 metric files are template-clones with the same shape. AI-fingerprint audit flagged this as a 4/5 AI-feel signal. Fix would collapse them into a registry pattern (`LLMJudgeMetric(name, template)`) — invasive structural change, deferred.

4. **Visibility**: Repo is PRIVATE per Rodolf's standing rule. README still calls it "Open-source Python toolkit" but that's aspirational; flip via `glab repo edit abanoub.rodolf/rag-eval-toolkit --visibility public` when ready.

## Files changed (audit/v1 vs main)
- 43 files changed
- 1,584 insertions
- 934 deletions
- Net: +650 lines, but largely test code (4 new test files: test_cli.py, test_backends.py, test_loaders.py extensions, test_report.py)

## URLs
- GitLab repo: https://gitlab.com/abanoub.rodolf/rag-eval-toolkit
- audit/v1 branch: https://gitlab.com/abanoub.rodolf/rag-eval-toolkit/-/tree/audit/v1
- Compare: https://gitlab.com/abanoub.rodolf/rag-eval-toolkit/-/compare/main...audit/v1
