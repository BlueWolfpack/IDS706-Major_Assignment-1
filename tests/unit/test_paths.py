from pathlib import Path
from typing import Callable

import pytest

from pipeline.paths import (
    ensure_data_dirs,
    get_data_dirs,
    get_features_dir,
    get_models_dir,
    get_predictions_dir,
    get_quality_dir,
    get_raw_dir,
)


@pytest.mark.unit
def test_get_data_dirs_resolves_all_expected_paths(tmp_path: Path) -> None:
    expected_data_dir = tmp_path.resolve() / "data"

    paths = get_data_dirs(tmp_path)

    assert paths == {
        "raw": expected_data_dir / "raw",
        "features": expected_data_dir / "features",
        "models": expected_data_dir / "models",
        "predictions": expected_data_dir / "predictions",
        "quality": expected_data_dir / "quality",
    }
    assert all(not path.exists() for path in paths.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path_helper", "directory_name"),
    [
        (get_raw_dir, "raw"),
        (get_features_dir, "features"),
        (get_models_dir, "models"),
        (get_predictions_dir, "predictions"),
        (get_quality_dir, "quality"),
    ],
)
def test_individual_path_helpers_resolve_under_data(
    path_helper: Callable[[str | Path | None], Path],
    directory_name: str,
    tmp_path: Path,
) -> None:
    assert path_helper(tmp_path) == tmp_path.resolve() / "data" / directory_name


@pytest.mark.unit
def test_ensure_data_dirs_is_idempotent(tmp_path: Path) -> None:
    first_result = ensure_data_dirs(tmp_path)
    second_result = ensure_data_dirs(tmp_path)

    assert second_result == first_result
    assert all(path.is_dir() for path in second_result.values())
