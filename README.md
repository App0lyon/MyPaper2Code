# MyPaper2Code

MyPaper2Code is a local CLI assistant for turning research papers into
traceable first implementation workspaces.

It is not a promise of automatic full reproduction. The current design is
agentic and human-in-the-loop: it extracts evidence, asks an LLM for structured
understanding, surfaces blocking ambiguities, records human decisions, creates a
paper-specific plan, generates a reviewed scaffold, validates it, and reports
fidelity.

```text
ingest paper
-> understand paper evidence
-> review ambiguities
-> decide / approve assumptions
-> plan implementation
-> implement
-> validate by level
-> report fidelity
```

## Capabilities

- Ingest PDF papers with PyMuPDF.
- Extract pages, sections, chunks, and heuristic artifacts for tables, figures,
  equations, and algorithms.
- Build Tantivy lexical search, NumPy vector search, and RRF hybrid retrieval.
- Use NVIDIA by default, fall back to Ollama, and fail if neither provider works.
- Produce rich `ResearchUnderstanding` artifacts with contributions,
  definitions, algorithms, equations, datasets, protocols, metrics,
  hyperparameters, resources, expected artifacts, evidence, and ambiguities.
- Block implementation when critical ambiguities are unresolved.
- Generate an agentic `ResearchImplementationPlan` instead of a fixed
  PyTorch-only plan.
- Write generated code under `generated/` and compatibility output under
  `generated_code/`.
- Validate with `smoke`, `contract`, or `repro` levels.
- Generate a fidelity report based on evidence, trace, validation, metrics, and
  protocol coverage.

## Installation

```bash
uv sync --all-extras
```

Useful checks:

```bash
uv run mypaper2code --help
uv run pytest
uv run ruff check .
```

## Workflow

From a PDF:

```bash
uv run mypaper2code ingest paper.pdf
uv run mypaper2code understand --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code review --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code decide --workspace workspaces/my_paper_20260523-120000 --id amb-001 --value "use the paper default"
uv run mypaper2code plan --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code approve-plan --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code implement --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code validate workspaces/my_paper_20260523-120000 --level contract
uv run mypaper2code report --workspace workspaces/my_paper_20260523-120000
```

One-command workflow:

```bash
uv run mypaper2code run paper.pdf \
  --provider nvidia \
  --dataset cifar10 \
  --level smoke \
  --ask-paper "What loss function is used?" \
  --ask-code "Where is the method implemented?"
```

## Workspace Layout

New workspaces are versioned by directory convention:

```text
workspaces/<paper_id_timestamp>/
├── paper/
│   ├── original.pdf
│   ├── pages.json
│   ├── extracted_sections.json
│   ├── chunks.json
│   ├── corpus_manifest.json
│   ├── tables/tables.json
│   ├── figures/figures.json
│   ├── equations/equations.json
│   ├── algorithms/algorithms.json
│   ├── tantivy_index/
│   ├── vectors.npy
│   └── vector_metadata.json
├── understanding/
│   ├── research_understanding.json
│   ├── review.md
│   └── provider_artifact.json
├── decisions/
│   ├── decisions.json
│   └── approvals.json
├── plan/
│   ├── implementation_plan.md
│   └── research_plan.json
├── generated/
├── validation/
├── trace/
└── metadata.json
```

Compatibility artifacts are also written to `analysis/`, `generated_code/`, and
`runs/` where useful.

## Commands

- `ingest paper.pdf`: create a workspace and retrieval indexes.
- `understand --workspace ...`: extract rich research understanding.
- `review --workspace ...`: list ambiguities and recorded decisions.
- `decide --workspace ... --id ... --value ...`: answer a blocking or
  non-blocking ambiguity.
- `approve-plan --workspace ...`: mark the current plan as approved.
- `approve-assumption --workspace ... --id ...`: approve an ambiguity without a
  full decision value.
- `plan --workspace ...`: create the agentic research implementation plan.
- `implement --workspace ...`: generate code only if blocking ambiguities are
  resolved.
- `validate WORKSPACE --level smoke|contract|repro`: run validation checks.
- `report --workspace ...`: write the fidelity report.
- `ask-paper` and `ask-code`: retrieve evidence from paper and generated code.

## Providers

Global configuration is stored in:

```text
~/.mypaper2code/config.json
```

NVIDIA is the default provider:

```bash
uv run mypaper2code config set provider nvidia
uv run mypaper2code config set model mistralai/mistral-medium-3.5-128b
uv run mypaper2code config set nvidia_api_key_env NVIDIA_API_KEY
```

Ollama is the fallback when NVIDIA is unavailable:

```bash
uv run mypaper2code config set provider ollama
uv run mypaper2code config set model llama3
uv run mypaper2code config set ollama_base_url http://localhost:11434
```

Test the provider chain:

```bash
uv run mypaper2code providers test
```

## Validation Levels

- `smoke`: imports, Ruff when available, and generated smoke tests.
- `contract`: smoke plus generated contract tests for method and protocol shape.
- `repro`: contract plus a toy reproduction command.

The fidelity score can be `blocked`, `low`, `medium`, `high`, or
`reproducible`. It never reports high fidelity based on trace presence alone;
evidence, method coverage, metrics, and protocol coverage are considered.

## Development

```bash
uv run ruff check .
uv run pytest
uv run mypaper2code --help
```

