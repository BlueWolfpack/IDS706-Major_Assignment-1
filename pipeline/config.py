from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data_dir: Path


def load_config(project_root: str | Path | None = None) -> PipelineConfig:
    """Load project configuration, optionally rooted at a test or alternate project path."""
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    return PipelineConfig(project_root=root, data_dir=root / "data")
