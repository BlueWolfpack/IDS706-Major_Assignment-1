from pathlib import Path

import pytest

from pipeline.config import PipelineConfig, load_config


@pytest.mark.unit
def test_load_config_uses_repository_root_by_default() -> None:
    config = load_config()

    assert isinstance(config, PipelineConfig)
    assert config.project_root == Path(__file__).resolve().parents[2]
    assert config.data_dir == config.project_root / "data"


@pytest.mark.unit
def test_load_config_accepts_an_alternate_project_root(tmp_path: Path) -> None:
    config = load_config(tmp_path / "nested" / "..")

    assert config.project_root == tmp_path.resolve()
    assert config.data_dir == tmp_path.resolve() / "data"
