"""Command-line interface for per-Area emissions regression analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__:
    from .analysis import load_data, run_analysis
else:
    from analysis import load_data, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit chronological agrofood-emissions regressions per Area."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source agrofood emissions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where metrics.csv and coefficients.csv will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input CSV does not exist: {args.input}")

    metrics, coefficients = run_analysis(load_data(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    coefficients_path = args.output_dir / "coefficients.csv"
    metrics.to_csv(metrics_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)

    successful = int(metrics["status"].isin(["success", "undefined_r2"]).sum())
    print(f"Evaluated {len(metrics)} Area/model combinations.")
    print(f"Fitted models: {successful}; not fitted: {len(metrics) - successful}.")
    print(f"Metrics: {metrics_path}")
    print(f"Coefficients: {coefficients_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
