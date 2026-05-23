from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from mypaper2code.core.io import load_model, write_model
from mypaper2code.core.models import PaperMetadata, WorkspaceMetadata
from mypaper2code.core.text import slugify


class WorkspaceManager:
    def __init__(self, base_dir: Path = Path("workspaces")) -> None:
        self.base_dir = base_dir

    def create(
        self,
        paper_path: Path,
        title: str,
        provider: str = "stub",
        model: str = "stub",
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        paper_id = slugify(title or paper_path.stem)
        workspace_id = f"{paper_id}_{timestamp}"
        root = self.base_dir / workspace_id

        for child in ("paper", "analysis", "generated_code", "runs"):
            (root / child).mkdir(parents=True, exist_ok=True)

        copied_pdf = root / "paper" / "original.pdf"
        shutil.copy2(paper_path, copied_pdf)
        metadata = WorkspaceMetadata(
            workspace_id=workspace_id,
            root=str(root),
            paper=PaperMetadata(
                paper_id=paper_id,
                title=title or paper_path.stem,
                source_path=str(copied_pdf),
            ),
            provider=provider,
            model=model,
        )
        write_model(root / "metadata.json", metadata)
        return root

    @staticmethod
    def load_metadata(workspace: Path) -> WorkspaceMetadata:
        return load_model(workspace / "metadata.json", WorkspaceMetadata)

    @staticmethod
    def ensure(workspace: Path) -> Path:
        if not (workspace / "metadata.json").exists():
            raise FileNotFoundError(f"Not a MyPaper2Code workspace: {workspace}")
        return workspace
