# MyPaper2Code

MyPaper2Code is a local CLI assistant for understanding AI research papers and
building a traceable first implementation workspace.

The project is intentionally not a one-shot "paper to perfect code" generator.
It follows an interactive workflow:

```text
ingest paper
-> ask questions about the paper
-> set implementation requirements
-> analyze the method
-> create a paper-specific implementation plan
-> implement that plan step by step
-> validate and report assumptions
-> ask questions about the generated implementation
```

The current MVP is implemented as a Python package managed by `uv`. It uses
file-based workspaces, JSON/YAML artifacts, Tantivy lexical search, local vector
search, and Reciprocal Rank Fusion (RRF).

## Current Capabilities

- Ingest a PDF paper with PyMuPDF.
- Extract text sections and chunks.
- Index paper chunks with:
  - Tantivy full-text search;
  - NumPy exact vector search;
  - RRF fusion over lexical and vector rankings.
- Ask sourced questions about the paper with page, section, passage, and score.
- Persist user requirements in the workspace.
- Analyze the paper into a structured method understanding.
- Generate a paper-specific implementation plan as Markdown and JSON.
- Implement the plan step by step into `generated_code/`.
- Write an implementation trace linking paper evidence, plan steps, files, and symbols.
- Validate generated code with import checks, Ruff when available, and pytest.
- Ask questions about the generated code by using the trace plus direct file search.
- Produce a fidelity and assumptions report.

## What This MVP Does Not Do Yet

- It does not provide a FastAPI API or web UI.
- It does not perform faithful full reproduction of arbitrary papers.
- It does not index generated code with RAG. Code questions use trace metadata and
  direct file inspection instead.
- NVIDIA and Ollama providers are available for LLM-backed paper understanding.
- NVIDIA is the default provider. If NVIDIA is unavailable, analysis falls back
  to Ollama. If Ollama is also unavailable, analysis fails instead of silently
  using a local heuristic.
- The implementation logic is plan-driven and paper-aware. `analyze` asks the
  selected provider for structured JSON understanding before planning.

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

## One-Command Workflow

The simplest entrypoint is `run`. It can start from a PDF or continue an
existing workspace.

From a PDF:

```bash
uv run mypaper2code run paper.pdf \
  --provider nvidia \
  --dataset cifar10 \
  --ask-paper "What loss function is used?" \
  --ask-code "Where is the loss implemented?"
```

From an existing workspace:

```bash
uv run mypaper2code run \
  --workspace workspaces/my_paper_20260523-120000 \
  --dataset cifar10
```

By default, `run` performs:

```text
ingest, when a PDF is provided
-> optional ask-paper
-> analyze
-> plan
-> implement
-> validate
-> report
-> optional ask-code
```

You can skip steps:

```bash
uv run mypaper2code run paper.pdf --no-validate
uv run mypaper2code run --workspace workspaces/my_paper_20260523-120000 --no-implement --no-report
```

## Advanced Step-by-Step CLI Workflow

### 1. Ingest a paper

```bash
uv run mypaper2code ingest paper.pdf
```

The command prints the created workspace path, for example:

```text
workspaces/my_paper_20260523-120000
```

Ingestion creates paper artifacts under `paper/`:

```text
paper/
├── original.pdf
├── extracted_sections.json
├── chunks.json
├── tantivy_index/
├── vectors.npy
├── vector_metadata.json
└── retrieval_config.json
```

### 2. Ask questions about the paper

```bash
uv run mypaper2code ask-paper "What loss function is used?" --workspace workspaces/my_paper_20260523-120000
```

`ask` is kept as a backward-compatible alias for `ask-paper`:

```bash
uv run mypaper2code ask "What datasets are used?" --workspace workspaces/my_paper_20260523-120000
```

### 3. Set implementation requirements

Requirements are stored in `analysis/requirements.yaml`.

```bash
uv run mypaper2code requirements set framework pytorch --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code requirements set dataset cifar10 --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code requirements set style research --workspace workspaces/my_paper_20260523-120000
uv run mypaper2code requirements get --workspace workspaces/my_paper_20260523-120000
```

Supported requirement fields are defined by `ImplementationRequirements`:

```yaml
framework: pytorch
dataset: cifar10
style: research
config_format: yaml
target_gpu_memory: null
implementation_level: minimal
include_tests: true
include_training_script: true
include_evaluation_script: true
provider: nvidia
model: mistralai/mistral-medium-3.5-128b
```

### 4. Analyze the paper

```bash
uv run mypaper2code analyze --workspace workspaces/my_paper_20260523-120000
```

This writes:

```text
analysis/method_summary.md
analysis/assumptions.md
analysis/paper_understanding.json
```

`paper_understanding.json` captures the current structured understanding of the
paper: architecture, loss, datasets, metrics, training hints, ambiguities, and
source passages.

### 5. Create the implementation plan

```bash
uv run mypaper2code plan --workspace workspaces/my_paper_20260523-120000
```

Optional overrides:

```bash
uv run mypaper2code plan \
  --workspace workspaces/my_paper_20260523-120000 \
  --framework pytorch \
  --dataset cifar10 \
  --style research
```

The plan is written to:

```text
analysis/implementation_plan.md
analysis/implementation_plan.json
```

Unlike a fixed template, the plan is represented as implementation steps. Each
step has a purpose, target files, symbols, paper source references, and
assumptions.

### 6. Implement the plan

```bash
uv run mypaper2code implement --workspace workspaces/my_paper_20260523-120000
```

`generate` is kept as a backward-compatible alias:

```bash
uv run mypaper2code generate --workspace workspaces/my_paper_20260523-120000
```

The implementation is written under:

```text
generated_code/
├── configs/
├── src/
├── scripts/
├── tests/
├── README.md
└── requirements.txt
```

The trace is written to:

```text
analysis/implementation_trace.json
```

This trace is the main source of truth for linking paper claims to generated
files and symbols.

### 7. Validate the generated code

```bash
uv run mypaper2code validate workspaces/my_paper_20260523-120000
```

Validation writes logs under `runs/`, including import checks, Ruff when
available, and pytest results.

### 8. Ask questions about the implementation

```bash
uv run mypaper2code ask-code "Where is the loss implemented?" --workspace workspaces/my_paper_20260523-120000
```

`ask-code` does not use a code RAG index. It uses:

- `analysis/implementation_trace.json`;
- direct search through generated source files;
- file names, symbols, and matched code lines.

This keeps the generated code itself as the source of truth.

### 9. Generate the fidelity report

```bash
uv run mypaper2code report --workspace workspaces/my_paper_20260523-120000
```

The report is written to:

```text
analysis/fidelity_report.md
```

It summarizes what was implemented, what remains ambiguous, and the current
fidelity level.

## Workspace Layout

A typical workspace looks like this:

```text
workspaces/
└── paper_name_timestamp/
    ├── paper/
    │   ├── original.pdf
    │   ├── extracted_sections.json
    │   ├── chunks.json
    │   ├── tantivy_index/
    │   ├── vectors.npy
    │   ├── vector_metadata.json
    │   └── retrieval_config.json
    ├── analysis/
    │   ├── requirements.yaml
    │   ├── method_summary.md
    │   ├── assumptions.md
    │   ├── paper_understanding.json
    │   ├── implementation_plan.md
    │   ├── implementation_plan.json
    │   ├── implementation_trace.json
    │   └── fidelity_report.md
    ├── generated_code/
    │   ├── configs/
    │   ├── src/
    │   ├── scripts/
    │   ├── tests/
    │   ├── README.md
    │   └── requirements.txt
    ├── runs/
    │   ├── imports.log
    │   ├── ruff.log
    │   └── pytest.log
    └── metadata.json
```

## Search Design

Paper retrieval is hybrid:

1. Tantivy retrieves full-text matches over `chunk_id`, `paper_id`, `section`,
   `page`, and `text`.
2. Vector search computes exact cosine similarity over chunk embeddings stored in
   `vectors.npy`.
3. RRF merges lexical and vector rankings:

```text
rrf_score = sum(1 / (rrf_k + rank))
```

If `sentence-transformers` is available, embeddings use
`sentence-transformers/all-MiniLM-L6-v2`. If not, the system falls back to a
deterministic hashing embedding so the MVP remains runnable locally.

## Providers

Global provider configuration is stored in:

```text
~/.mypaper2code/config.json
```

Commands:

```bash
uv run mypaper2code config set provider nvidia
uv run mypaper2code config set model mistralai/mistral-medium-3.5-128b
uv run mypaper2code config get
```

Provider interfaces currently exist for:

- `ollama`
- `nvidia`

`nvidia` is the default provider. When NVIDIA is selected and unavailable,
MyPaper2Code tries Ollama next. If Ollama is unavailable too, the command fails.

### Ollama

Ollama uses the local chat API:

```text
POST http://localhost:11434/api/chat
```

Configure it:

```bash
uv run mypaper2code config set provider ollama
uv run mypaper2code config set model llama3
uv run mypaper2code config set ollama_base_url http://localhost:11434
```

Then test it:

```bash
uv run mypaper2code providers test
```

The selected Ollama model must already be available locally, for example through:

```bash
ollama pull llama3
```

### NVIDIA

NVIDIA uses the OpenAI-compatible NIM endpoint:

```text
POST https://integrate.api.nvidia.com/v1/chat/completions
```

Configure it:

```bash
uv run mypaper2code config set provider nvidia
uv run mypaper2code config set model mistralai/mistral-medium-3.5-128b
uv run mypaper2code config set nvidia_base_url https://integrate.api.nvidia.com/v1
uv run mypaper2code config set nvidia_api_key_env NVIDIA_API_KEY
```

Set your API key in the environment or in a local `.env` file. The `.env` file is
loaded automatically and does not override variables already present in the
environment.

```bash
$env:NVIDIA_API_KEY="your-api-key"
```

Example `.env`:

```bash
NVIDIA_API_KEY="your-api-key"
```

Then test it:

```bash
uv run mypaper2code providers test
```

You can override provider and model for a single test:

```bash
uv run mypaper2code providers test --provider ollama --model llama3
uv run mypaper2code providers test --provider nvidia --model mistralai/mistral-medium-3.5-128b
```

When `analyze` runs with `provider=ollama` or `provider=nvidia`, MyPaper2Code
uses the selected model to extract structured implementation facts from retrieved
paper passages. If NVIDIA fails, Ollama is tried as the fallback. If the selected
provider chain fails or returns invalid JSON, the command fails.

## Internal Modules

Main package areas:

```text
src/mypaper2code/
├── cli.py
├── core/
│   ├── io.py
│   ├── models.py
│   └── text.py
├── providers/
│   └── base.py
└── services/
    ├── analysis.py
    ├── code_qa.py
    ├── config.py
    ├── generation.py
    ├── ingestion.py
    ├── planning.py
    ├── report.py
    ├── requirements.py
    ├── validation.py
    ├── workspace.py
    └── search/
        ├── hybrid.py
        ├── rrf.py
        ├── tantivy_index.py
        └── vector_index.py
```

## Development

Run the full checks:

```bash
uv run pytest
uv run ruff check .
uv run mypaper2code --help
```

Current test coverage includes:

- PDF extraction and chunking;
- workspace creation;
- Tantivy/vector index creation and reload;
- RRF deduplication;
- CLI commands;
- requirements persistence;
- plan-driven implementation;
- implementation trace;
- code question answering;
- fidelity report;
- validation of generated code.
- Ollama and NVIDIA provider payload construction without making network calls.

## Roadmap

Next useful implementation steps:

1. Improve LLM prompts and schema validation for paper understanding.
2. Add LLM-assisted implementation step generation with stricter review gates.
3. Add richer table/figure extraction.
4. Improve plan execution with per-step validation and repair.
5. Add stronger AST-based code inspection for `ask-code`.
6. Add optional API/server mode after the CLI workflow is stable.
