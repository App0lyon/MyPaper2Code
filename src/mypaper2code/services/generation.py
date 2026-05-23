from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import load_model, write_json, write_yaml
from mypaper2code.core.models import (
    CodeLocation,
    ImplementationPlan,
    ImplementationRequirements,
    ImplementationTrace,
    ImplementationTraceEntry,
    SourceSpan,
)
from mypaper2code.services.planning import ImplementationPlanner


class CodeWriter:
    def implement(self, workspace: Path) -> Path:
        plan_path = workspace / "analysis" / "implementation_plan.json"
        if plan_path.exists():
            plan = load_model(plan_path, ImplementationPlan)
        else:
            plan = ImplementationPlanner().create_plan(workspace, ImplementationRequirements())

        root = workspace / "generated_code"
        root.mkdir(parents=True, exist_ok=True)
        trace_entries: list[ImplementationTraceEntry] = []
        for step in plan.steps:
            written = self._implement_step(root, plan, step.step_id)
            trace_entries.append(
                ImplementationTraceEntry(
                    step_id=step.step_id,
                    paper_claim=step.purpose,
                    source=step.source_refs[0] if step.source_refs else _first_source(plan),
                    implemented_in=[
                        CodeLocation(
                            file=relative,
                            symbol=symbol,
                            line_start=_line_for_symbol(root / relative, symbol),
                        )
                        for relative, symbol in written
                    ],
                    assumptions=step.assumptions,
                )
            )

        write_json(
            workspace / "analysis" / "implementation_trace.json",
            ImplementationTrace(entries=trace_entries).model_dump(mode="json"),
        )
        return root

    def generate(self, workspace: Path) -> Path:
        return self.implement(workspace)

    def _implement_step(
        self,
        root: Path,
        plan: ImplementationPlan,
        step_id: str,
    ) -> list[tuple[str, str | None]]:
        req = plan.requirements
        understanding = plan.understanding
        model_class = _model_class(understanding.architecture)
        loss_class = _loss_class(understanding.loss)
        if step_id == "config":
            return self._write_config(root, plan, model_class)
        if step_id == "data":
            return self._write_many(root, {"src/data/datasets.py": _data_code(req.dataset)})
        if step_id == "model":
            return self._write_many(
                root,
                {"src/models/model.py": _model_code(model_class, understanding.architecture)},
            )
        if step_id == "loss":
            return self._write_many(
                root,
                {"src/losses/losses.py": _loss_code(loss_class, understanding.loss)},
            )
        if step_id == "training":
            return self._write_many(
                root,
                {
                    "src/training/trainer.py": _trainer_code(),
                    "src/evaluation/metrics.py": _metrics_code(),
                    "src/utils/config.py": _config_code(),
                    "scripts/train.py": _train_script_code(model_class),
                    "scripts/evaluate.py": _evaluate_script_code(model_class),
                },
            )
        if step_id == "tests":
            return self._write_many(root, {"tests/test_smoke.py": _test_code(model_class)})
        return []

    def _write_config(
        self,
        root: Path,
        plan: ImplementationPlan,
        model_class: str,
    ) -> list[tuple[str, str | None]]:
        req = plan.requirements
        _ensure_packages(root)
        write_yaml(
            root / "configs" / "default.yaml",
            {
                "dataset": req.dataset,
                "paper_understanding": plan.understanding.model_dump(mode="json"),
                "model": {
                    "class_name": model_class,
                    "architecture": plan.understanding.architecture,
                    "input_dim": 3072,
                    "hidden_dim": 256,
                    "num_classes": 10,
                },
                "loss": {"name": plan.understanding.loss},
                "training": {
                    "batch_size": 32,
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "optimizer": "adamw",
                },
            },
        )
        readme = _readme(plan)
        files = {
            "requirements.txt": "torch\npyyaml\n",
            "README.md": readme,
        }
        written = self._write_many(root, files)
        written.append(("configs/default.yaml", None))
        return written

    @staticmethod
    def _write_many(root: Path, files: dict[str, str]) -> list[tuple[str, str | None]]:
        _ensure_packages(root)
        written: list[tuple[str, str | None]] = []
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written.append((relative, _primary_symbol(content)))
        return written


def _ensure_packages(root: Path) -> None:
    for directory in (
        "src",
        "src/data",
        "src/models",
        "src/losses",
        "src/training",
        "src/evaluation",
        "src/utils",
        "scripts",
        "tests",
        "configs",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for package in (
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/losses/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
    ):
        (root / package).touch()


def _model_class(architecture: str) -> str:
    return {
        "transformer": "PaperTransformer",
        "cnn": "PaperCNN",
        "diffusion": "PaperDiffusionStub",
        "gan": "PaperGANStub",
        "vae": "PaperVAE",
        "mlp": "PaperMLP",
    }.get(architecture, "PaperModel")


def _loss_class(loss: str) -> str:
    return {
        "contrastive": "ContrastiveLoss",
        "cross_entropy": "PaperCrossEntropyLoss",
        "mse": "PaperMSELoss",
        "triplet": "TripletLoss",
    }.get(loss, "PaperLoss")


def _data_code(dataset: str) -> str:
    return f'''from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_data_loader(
    batch_size: int = 32,
    input_dim: int = 3072,
    num_classes: int = 10,
    dataset_name: str = "{dataset}",
) -> DataLoader:
    """Return a local smoke-test loader matching the requested dataset contract."""
    del dataset_name
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)
'''


def _model_code(class_name: str, architecture: str) -> str:
    if architecture == "transformer":
        body = f'''class {class_name}(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(x.view(x.size(0), -1)).unsqueeze(1)
        encoded = self.encoder(tokens).squeeze(1)
        return self.head(encoded)
'''
    elif architecture == "cnn":
        body = f'''class {class_name}(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        del input_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x.view(x.size(0), 3, 32, 32)
        return self.head(self.features(image))
'''
    else:
        body = f'''class {class_name}(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(x.size(0), -1))
'''
    return f'''from __future__ import annotations

import torch
from torch import nn


{body}
'''


def _loss_code(class_name: str, loss: str) -> str:
    if loss == "contrastive":
        class_body = f'''class {class_name}(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scaled = logits / self.temperature
        return nn.functional.cross_entropy(scaled, target)
'''
    elif loss == "mse":
        class_body = f'''class {class_name}(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        one_hot = nn.functional.one_hot(target, num_classes=logits.size(-1)).float()
        return nn.functional.mse_loss(logits, one_hot)
'''
    elif loss == "triplet":
        class_body = f'''class {class_name}(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return logits.pow(2).mean()
'''
    else:
        class_body = f'''class {class_name}(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target)
'''
    return f'''from __future__ import annotations

import torch
from torch import nn


{class_body}


def build_loss() -> nn.Module:
    return {class_name}()
'''


def _trainer_code() -> str:
    return '''from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)
'''


def _metrics_code() -> str:
    return '''from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == target).float().mean())
'''


def _config_code() -> str:
    return '''from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
'''


def _train_script_code(model_class: str) -> str:
    return f'''from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.datasets import make_data_loader  # noqa: E402
from src.losses.losses import build_loss  # noqa: E402
from src.models.model import {model_class}  # noqa: E402
from src.training.trainer import train_one_epoch  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    model = {model_class}(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
    )
    loader = make_data_loader(
        config["training"]["batch_size"],
        config["model"]["input_dim"],
        config["model"]["num_classes"],
        config["dataset"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    loss = train_one_epoch(model, loader, optimizer, build_loss())
    print(f"loss={{loss:.4f}}")


if __name__ == "__main__":
    main()
'''


def _evaluate_script_code(model_class: str) -> str:
    return f'''from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import make_data_loader  # noqa: E402
from src.evaluation.metrics import accuracy  # noqa: E402
from src.models.model import {model_class}  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    model = {model_class}(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
    )
    loader = make_data_loader(
        config["training"]["batch_size"],
        config["model"]["input_dim"],
        config["model"]["num_classes"],
        config["dataset"],
    )
    x, y = next(iter(loader))
    print(f"accuracy={{accuracy(model(x), y):.4f}}")


if __name__ == "__main__":
    main()
'''


def _test_code(model_class: str) -> str:
    return f'''from __future__ import annotations

from src.data.datasets import make_data_loader
from src.evaluation.metrics import accuracy
from src.losses.losses import build_loss
from src.models.model import {model_class}


def test_generated_pipeline_smoke() -> None:
    loader = make_data_loader(batch_size=4)
    x, y = next(iter(loader))
    model = {model_class}()
    logits = model(x)
    loss = build_loss()(logits, y)
    assert logits.shape == (4, 10)
    assert loss.item() >= 0.0
    assert 0.0 <= accuracy(logits, y) <= 1.0
'''


def _readme(plan: ImplementationPlan) -> str:
    sources = "\n".join(
        f"- Page {source.page}, section `{source.section}`: {source.text}"
        for source in plan.sources
    )
    assumptions = "\n".join(f"- {item}" for item in plan.assumptions)
    steps = "\n".join(
        f"- `{step.step_id}`: {step.title} -> {', '.join(step.files)}" for step in plan.steps
    )
    return f"""# Generated Paper Implementation

This implementation was created from a paper-specific implementation plan.

## Paper Understanding

- Architecture: `{plan.understanding.architecture}`
- Loss: `{plan.understanding.loss}`
- Datasets: `{", ".join(plan.understanding.datasets) or "unspecified"}`
- Metrics: `{", ".join(plan.understanding.metrics) or "unspecified"}`

## Plan Steps

{steps}

## Run

```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
pytest
```

## Assumptions

{assumptions}

## Paper Sources

{sources or "- No source passage available."}
"""


def _primary_symbol(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("class "):
            return stripped.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0]
        if stripped.startswith("def "):
            return stripped.split("def ", 1)[1].split("(", 1)[0]
    return None


def _line_for_symbol(path: Path, symbol: str | None) -> int | None:
    if not symbol or not path.exists():
        return None
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if f"class {symbol}" in line or f"def {symbol}" in line:
            return idx
    return None


def _first_source(plan: ImplementationPlan) -> SourceSpan | None:
    return plan.sources[0] if plan.sources else None
