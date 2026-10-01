from pathlib import Path

import pytest

from pipeline.paths import get_data_dirs


@pytest.mark.regression
def test_data_paths_remain_under_the_selected_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    paths = get_data_dirs(project_root)
    expected_data_dir = project_root.resolve() / "data"

    assert all(path.parent == expected_data_dir for path in paths.values())
