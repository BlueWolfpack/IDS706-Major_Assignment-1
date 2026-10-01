from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.paths import ensure_data_dirs, get_data_dirs


@pytest.mark.integration
def test_config_and_directory_creation_work_together(tmp_path: Path) -> None:
    config = load_config(tmp_path)

    created_dirs = ensure_data_dirs(config.project_root)

    assert created_dirs == get_data_dirs(config.project_root)
    assert set(created_dirs) == {"raw", "features", "models", "predictions", "quality"}
    assert all(path.is_dir() for path in created_dirs.values())
