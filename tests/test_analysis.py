from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import MODEL_SPECS, run_analysis, split_chronologically
from src.cli import main


def make_data(
    *,
    area: str = "A",
    count: int = 60,
    constant_temperature: bool = False,
) -> pd.DataFrame:
    years = np.arange(1960, 1960 + count)
    temperature = (
        np.full(count, 10.0)
        if constant_temperature
        else 10.0 + np.sin(np.arange(count) * 0.7)
    )
    return pd.DataFrame(
        {
            "Area": area,
            "Year": years,
            "Average Temperature °C": temperature,
            "Food Transport": np.arange(count, dtype=float) * 1.5,
            "total_emission": 2.5 * years + 7.0 * temperature + 100.0,
        }
    )


def test_chronological_split_keeps_latest_years_for_test() -> None:
    observations = make_data(count=10).sample(frac=1, random_state=17)

    train, test = split_chronologically(observations)

    assert train["Year"].max() < test["Year"].min()
    assert test["Year"].tolist() == [1968, 1969]


def test_time_climate_recovers_signal_and_uses_canonical_feature() -> None:
    data = make_data()

    metrics, coefficients = run_analysis(data)
    result = metrics.loc[metrics["model"].eq("time_climate")].iloc[0]
    fitted = coefficients.loc[coefficients["model"].eq("time_climate")]
    by_predictor = fitted.set_index("predictor")

    assert result["status"] == "success"
    assert result["train_observations"] == 48
    assert result["test_observations"] == 12
    assert result["RMSE"] == pytest.approx(0, abs=1e-8)
    assert result["R2"] == pytest.approx(1, abs=1e-12)
    assert by_predictor.loc["Year", "coefficient"] == pytest.approx(2.5)
    assert by_predictor.loc["average_temperature", "coefficient"] == pytest.approx(7)
    train = data.iloc[:48]
    expected_standardized_temperature = (
        7
        * train["Average Temperature °C"].std(ddof=0)
        / train["total_emission"].std(ddof=0)
    )
    assert by_predictor.loc[
        "average_temperature", "standardized_coefficient"
    ] == pytest.approx(expected_standardized_temperature)
    assert "Area" not in result["predictors"]
    assert "total_emission" not in result["predictors"]
    assert "Average Temperature °C" not in result["predictors"]


def test_each_area_is_fitted_independently() -> None:
    first = make_data(area="A")
    second = make_data(area="B")
    second["total_emission"] += 500
    data = pd.concat([first, second], ignore_index=True)

    metrics, coefficients = run_analysis(data)

    time_metrics = metrics.loc[metrics["model"].eq("time_climate")]
    assert time_metrics["Area"].tolist() == ["A", "B"]
    assert len(time_metrics) == 2
    intercepts = coefficients.loc[
        coefficients["model"].eq("time_climate")
        & coefficients["predictor"].eq("Intercept")
    ].set_index("Area")["coefficient"]
    assert intercepts["B"] - intercepts["A"] == pytest.approx(500)


def test_constant_predictors_are_reported_and_retained() -> None:
    data = make_data(count=60, constant_temperature=True)

    metrics, coefficients = run_analysis(data)
    result = metrics.loc[metrics["model"].eq("time_climate")].iloc[0]
    temperature = coefficients.loc[
        coefficients["model"].eq("time_climate")
        & coefficients["predictor"].eq("average_temperature")
    ].iloc[0]

    assert result["status"] == "success"
    assert result["constant_predictors"] == "average_temperature"
    assert temperature["coefficient"] == pytest.approx(0)
    assert temperature["standardized_coefficient"] == pytest.approx(0)


def test_insufficient_training_observations_are_retained_with_reason() -> None:
    metrics, coefficients = run_analysis(make_data(count=10))
    result = metrics.loc[metrics["model"].eq("time_climate")].iloc[0]

    assert result["status"] == "insufficient_observations"
    assert "Need at least" in result["reason"]
    assert result["train_observations"] == 8
    assert result["test_observations"] == 2
    assert coefficients.loc[coefficients["model"].eq("time_climate")].empty


def test_constant_test_target_keeps_rmse_and_undefined_r2() -> None:
    data = make_data()
    data.loc[data.index[-12:], "total_emission"] = 50.0

    metrics, _ = run_analysis(data)
    result = metrics.loc[metrics["model"].eq("time_climate")].iloc[0]

    assert result["status"] == "undefined_r2"
    assert np.isfinite(result["RMSE"])
    assert np.isnan(result["R2"])
    assert "held-out target is constant" in result["reason"]


def test_missing_values_change_only_the_affected_model_sample() -> None:
    data = make_data()
    data["Food Processing"] = np.arange(len(data), dtype=float) * 2
    data["Food Packaging"] = np.arange(len(data), dtype=float) * 3
    data.loc[0, "Average Temperature °C"] = np.nan

    metrics, _ = run_analysis(data)
    by_model = metrics.set_index("model")

    assert by_model.loc["time_climate", "observations"] == 59
    assert by_model.loc["time_climate", "train_observations"] == 47
    assert by_model.loc["time_climate", "test_observations"] == 12
    assert by_model.loc["food_chain_upstream", "observations"] == 60
    assert by_model.loc["food_chain_upstream", "train_observations"] == 48
    assert by_model.loc["food_chain_upstream", "test_observations"] == 12


@pytest.mark.parametrize(
    "temperature_header",
    [
        "Average Temperature °C",
        "Average Temperature Â°C",
        "average_temperature",
    ],
)
def test_recognized_temperature_headers_are_normalized(
    temperature_header: str,
) -> None:
    data = make_data().rename(
        columns={"Average Temperature °C": temperature_header}
    )

    metrics, coefficients = run_analysis(data)
    result = metrics.loc[metrics["model"].eq("time_climate")].iloc[0]

    assert result["status"] == "success"
    assert "average_temperature" in result["predictors"]
    assert "average_temperature" in set(
        coefficients.loc[
            coefficients["model"].eq("time_climate"), "predictor"
        ]
    )


@pytest.mark.parametrize(
    "duplicate_header",
    ["average_temperature", "Average Temperature Â°C"],
)
def test_multiple_recognized_temperature_headers_fail_clearly(
    duplicate_header: str,
) -> None:
    data = make_data()
    data[duplicate_header] = data["Average Temperature °C"]

    with pytest.raises(ValueError, match="multiple recognized temperature columns"):
        run_analysis(data)


def test_non_numeric_values_are_coerced_without_mutating_input() -> None:
    data = make_data()
    data["Average Temperature °C"] = data["Average Temperature °C"].astype(object)
    data.loc[0, "Average Temperature °C"] = "not numeric"
    original = data.copy(deep=True)

    metrics, _ = run_analysis(data)
    result = metrics.loc[metrics["model"].eq("time_climate")].iloc[0]

    assert result["observations"] == 59
    pd.testing.assert_frame_equal(data, original)


def test_required_columns_fail_clearly() -> None:
    data = make_data().drop(columns="total_emission")

    with pytest.raises(ValueError, match="total_emission"):
        run_analysis(data)


def test_partial_schema_retains_all_models_and_reports_missing_fields() -> None:
    data = make_data()

    metrics, coefficients = run_analysis(data)

    assert len(metrics) == len(MODEL_SPECS)
    assert metrics["Area"].eq("A").all()
    assert metrics["model"].nunique() == len(MODEL_SPECS)
    time_status = metrics.loc[
        metrics["model"].eq("time_climate"), "status"
    ].item()
    combined_status = metrics.loc[
        metrics["model"].eq("combined"), "status"
    ].item()
    assert time_status == "success"
    assert combined_status == "success"
    assert "missing_predictors" in set(metrics["status"])
    missing = metrics.loc[metrics["status"].eq("missing_predictors")]
    assert missing["unavailable_predictors"].str.len().gt(0).all()
    assert metrics.columns.tolist() == [
        "Area",
        "model",
        "predictors",
        "observations",
        "train_observations",
        "test_observations",
        "RMSE",
        "R2",
        "status",
        "reason",
        "unavailable_predictors",
        "constant_predictors",
        "max_abs_correlation",
        "high_collinearity",
    ]
    assert not coefficients.empty
    assert coefficients["Area"].eq("A").all()
    assert "average_temperature" in set(coefficients["predictor"])


def test_cli_writes_metrics_and_coefficients(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic.csv"
    make_data().to_csv(dataset_path, index=False)
    output_dir = tmp_path / "results"

    result = main(["--input", str(dataset_path), "--output-dir", str(output_dir)])

    metrics = pd.read_csv(output_dir / "metrics.csv")
    coefficients = pd.read_csv(output_dir / "coefficients.csv")
    assert result == 0
    assert len(metrics) == len(MODEL_SPECS)
    assert {"RMSE", "R2", "status", "reason"}.issubset(metrics.columns)
    assert {"coefficient", "standardized_coefficient"}.issubset(coefficients.columns)


def test_missing_input_path_is_reported_by_cli(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--input",
                str(tmp_path / "absent.csv"),
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )

    assert error.value.code == 2
