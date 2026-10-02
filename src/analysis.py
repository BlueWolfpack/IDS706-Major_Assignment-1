"""Prepare and evaluate chronological per-Area regression models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "total_emission"
AREA = "Area"
YEAR = "Year"
TEMPERATURE = "average_temperature"
# why is TEST_FRACTION set to 0.2?
TEST_FRACTION = 0.2
MIN_TRAIN_SAMPLES_PER_PARAMETER = 5
COLLINEARITY_THRESHOLD = 0.95

# is this part necessary?
TEMPERATURE_HEADERS = (
    "Average Temperature °C",
    "Average Temperature Â°C",
    TEMPERATURE,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    predictors: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec("time_climate", (YEAR, TEMPERATURE)),
    ModelSpec(
        "population_climate",
        (
            YEAR,
            TEMPERATURE,
            "Rural population",
            "Urban population",
            "Total Population - Male",
            "Total Population - Female",
        ),
    ),
    ModelSpec(
        "population_climate_sensitivity",
        (YEAR, TEMPERATURE, "Rural population", "Urban population"),
    ),
    ModelSpec(
        "land_fire",
        (
            YEAR,
            "Forestland",
            "Net Forest conversion",
            "Forest fires",
            "Fires in organic soils",
            "Fires in humid tropical forests",
        ),
    ),
    ModelSpec(
        "agriculture_inputs",
        (
            YEAR,
            "Crop Residues",
            "Rice Cultivation",
            "Pesticides Manufacturing",
            "Fertilizers Manufacturing",
        ),
    ),
    ModelSpec(
        "fire_and_soils",
        (
            YEAR,
            "Savanna fires",
            "Forest fires",
            "Drained organic soils (CO2)",
            "Fires in organic soils",
            "Fires in humid tropical forests",
        ),
    ),
    ModelSpec(
        "food_chain_upstream",
        (YEAR, "Food Transport", "Food Processing", "Food Packaging"),
    ),
    ModelSpec(
        "food_chain_downstream",
        (
            YEAR,
            "Food Household Consumption",
            "Food Retail",
            "Agrifood Systems Waste Disposal",
        ),
    ),
    ModelSpec(
        "livestock_manure",
        (
            YEAR,
            "Manure applied to Soils",
            "Manure left on Pasture",
            "Manure Management",
        ),
    ),
    ModelSpec(
        "farm_energy_industry",
        (YEAR, "On-farm Electricity Use", "On-farm energy use", "IPPU"),
    ),
    ModelSpec("combined", (YEAR, TEMPERATURE, "Food Transport")),
)

METRICS_COLUMNS = (
    AREA,
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
)
COEFFICIENT_COLUMNS = (
    AREA,
    "model",
    "predictor",
    "coefficient",
    "standardized_coefficient",
)


def canonicalize_temperature_column(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the source temperature header to the documented internal name."""
    source_columns = [name for name in TEMPERATURE_HEADERS if name in data.columns]
    if len(source_columns) > 1:
        raise ValueError(
            "Input contains multiple recognized temperature columns; "
            "rename or remove the duplicate."
        )

    result = data.copy()
    if source_columns and source_columns[0] != TEMPERATURE:
        result = result.rename(columns={source_columns[0]: TEMPERATURE})
    return result


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate identifiers and coerce available modeling columns to numeric."""
    prepared = canonicalize_temperature_column(data)
    required = {AREA, YEAR, TARGET}
    missing = sorted(required.difference(prepared.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    prepared[AREA] = prepared[AREA].astype("string").str.strip()
    prepared[YEAR] = pd.to_numeric(prepared[YEAR], errors="coerce")
    numeric_columns = {
        column
        for spec in MODEL_SPECS
        for column in spec.predictors
        if column in prepared.columns and column != AREA
    }
    numeric_columns.add(TARGET)
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=[AREA, YEAR]).copy()
    prepared = prepared.loc[prepared[AREA].ne("")].copy()
    if prepared.empty:
        raise ValueError("Input has no rows with a valid Area and Year.")
    return prepared


def split_chronologically(
    observations: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sorted observations so the final years are reserved for testing."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    ordered = observations.sort_values(YEAR, kind="stable")
    test_count = int(np.ceil(len(ordered) * test_fraction))
    split_at = len(ordered) - test_count
    return ordered.iloc[:split_at], ordered.iloc[split_at:]


def _correlation_diagnostic(training_features: pd.DataFrame) -> float:
    if training_features.shape[1] < 2:
        return 0.0
    correlations = training_features.corr().abs().to_numpy()
    upper_triangle = correlations[np.triu_indices_from(correlations, k=1)]
    finite_correlations = upper_triangle[np.isfinite(upper_triangle)]
    if finite_correlations.size == 0:
        return 0.0
    return float(finite_correlations.max())


def _status_row(
    area: str,
    spec: ModelSpec,
    status: str,
    reason: str,
    *,
    observations: int = 0,
    train_observations: int = 0,
    test_observations: int = 0,
    unavailable_predictors: Iterable[str] = (),
    constant_predictors: Iterable[str] = (),
    max_abs_correlation: float = np.nan,
) -> dict[str, object]:
    return {
        AREA: area,
        "model": spec.name,
        "predictors": ";".join(spec.predictors),
        "observations": observations,
        "train_observations": train_observations,
        "test_observations": test_observations,
        "RMSE": np.nan,
        "R2": np.nan,
        "status": status,
        "reason": reason,
        "unavailable_predictors": ";".join(unavailable_predictors),
        "constant_predictors": ";".join(constant_predictors),
        "max_abs_correlation": max_abs_correlation,
        "high_collinearity": (
            max_abs_correlation >= COLLINEARITY_THRESHOLD
            if np.isfinite(max_abs_correlation)
            else False
        ),
    }


def _fit_area_model(
    area: str,
    area_data: pd.DataFrame,
    spec: ModelSpec,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    unavailable = [name for name in spec.predictors if name not in area_data.columns]
    if unavailable:
        return (
            _status_row(
                area,
                spec,
                "missing_predictors",
                f"Input is missing planned predictors: {', '.join(unavailable)}.",
                unavailable_predictors=unavailable,
            ),
            [],
        )

    usable = area_data.loc[:, [*spec.predictors, TARGET]].dropna()
    train, test = split_chronologically(usable)
    n_observations = len(usable)
    n_train = len(train)
    n_test = len(test)
    if n_test < 2 or n_train < MIN_TRAIN_SAMPLES_PER_PARAMETER * (
        len(spec.predictors) + 1
    ):
        minimum_train = MIN_TRAIN_SAMPLES_PER_PARAMETER * (len(spec.predictors) + 1)
        return (
            _status_row(
                area,
                spec,
                "insufficient_observations",
                f"Need at least {minimum_train} training observations and 2 test "
                f"observations; available split is {n_train} train / {n_test} test.",
                observations=n_observations,
                train_observations=n_train,
                test_observations=n_test,
            ),
            [],
        )

    x_train = train.loc[:, spec.predictors]
    y_train = train[TARGET]
    x_test = test.loc[:, spec.predictors]
    y_test = test[TARGET]
    train_target_std = float(y_train.std(ddof=0))
    constant_predictors = [
        name for name in spec.predictors if x_train[name].nunique(dropna=True) <= 1
    ]
    max_abs_correlation = _correlation_diagnostic(x_train)

    if train_target_std == 0:
        return (
            _status_row(
                area,
                spec,
                "constant_training_target",
                "The training target is constant, so this model has no estimable "
                "target variation.",
                observations=n_observations,
                train_observations=n_train,
                test_observations=n_test,
                constant_predictors=constant_predictors,
                max_abs_correlation=max_abs_correlation,
            ),
            [],
        )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    test_target_is_constant = float(y_test.std(ddof=0)) == 0
    r2 = np.nan if test_target_is_constant else float(r2_score(y_test, predictions))
    status = "undefined_r2" if test_target_is_constant else "success"
    reason = (
        "R2 is undefined because the held-out target is constant."
        if test_target_is_constant
        else ""
    )
    metrics = _status_row(
        area,
        spec,
        status,
        reason,
        observations=n_observations,
        train_observations=n_train,
        test_observations=n_test,
        constant_predictors=constant_predictors,
        max_abs_correlation=max_abs_correlation,
    )
    metrics["RMSE"] = rmse
    metrics["R2"] = r2

    scaler = pipeline.named_steps["scale"]
    regression = pipeline.named_steps["regression"]
    raw_coefficients = regression.coef_ / scaler.scale_
    coefficients = [
        {
            AREA: area,
            "model": spec.name,
            "predictor": name,
            "coefficient": float(raw_coefficients[index]),
            "standardized_coefficient": float(
                regression.coef_[index] / train_target_std
            ),
        }
        for index, name in enumerate(spec.predictors)
    ]
    raw_intercept = float(
        regression.intercept_ - np.dot(raw_coefficients, scaler.mean_)
    )
    coefficients.append(
        {
            AREA: area,
            "model": spec.name,
            "predictor": "Intercept",
            "coefficient": raw_intercept,
            "standardized_coefficient": np.nan,
        }
    )
    return metrics, coefficients


def run_analysis(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every declared model for every Area in the input dataframe."""
    prepared = prepare_data(data)
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for area, area_data in prepared.groupby(AREA, sort=True, observed=True):
        for spec in MODEL_SPECS:
            metrics, coefficients = _fit_area_model(str(area), area_data, spec)
            metric_rows.append(metrics)
            coefficient_rows.extend(coefficients)

    metrics_frame = pd.DataFrame(metric_rows, columns=METRICS_COLUMNS)
    coefficients_frame = pd.DataFrame(coefficient_rows, columns=COEFFICIENT_COLUMNS)
    metrics_frame = metrics_frame.sort_values(
        [AREA, "RMSE"],
        ascending=[True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return metrics_frame, coefficients_frame


def load_data(input_path: str | Path) -> pd.DataFrame:
    """Read a CSV while raising a direct error when its path is unavailable."""
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")
