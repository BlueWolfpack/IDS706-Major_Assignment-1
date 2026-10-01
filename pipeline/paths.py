from pathlib import Path

from pipeline.config import load_config


def get_data_dirs(project_root: str | Path | None = None) -> dict[str, Path]:
    data_dir = load_config(project_root).data_dir
    return {
        "raw": data_dir / "raw",
        "features": data_dir / "features",
        "models": data_dir / "models",
        "predictions": data_dir / "predictions",
        "quality": data_dir / "quality",
    }


def get_raw_dir(project_root: str | Path | None = None) -> Path:
    return get_data_dirs(project_root)["raw"]


def get_features_dir(project_root: str | Path | None = None) -> Path:
    return get_data_dirs(project_root)["features"]


def get_models_dir(project_root: str | Path | None = None) -> Path:
    return get_data_dirs(project_root)["models"]


def get_predictions_dir(project_root: str | Path | None = None) -> Path:
    return get_data_dirs(project_root)["predictions"]


def get_quality_dir(project_root: str | Path | None = None) -> Path:
    return get_data_dirs(project_root)["quality"]


def ensure_data_dirs(project_root: str | Path | None = None) -> dict[str, Path]:
    directories = get_data_dirs(project_root)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories
