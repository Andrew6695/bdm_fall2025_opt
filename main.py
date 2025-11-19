"""Command-line interface for the portfolio optimization pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.portfolio_pipeline import (
    IPOPT_PATH,
    evaluate_out_of_sample,
    run_portfolio_example,
    select_max_return_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean-variance efficient frontier sweep using Pyomo + IPOPT.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["GE", "KO", "NVDA"],
        help="List of ticker symbols (space separated).",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2024-01-01",
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--ipopt-path",
        default=IPOPT_PATH,
        help="Path to the IPOPT executable (installed via idaes get-extensions).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=200,
        help="Number of variance caps to sweep across the frontier.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to store generated figures and evaluation tables.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying matplotlib windows (plots are still saved to disk).",
    )
    parser.add_argument(
        "--eval-start",
        help="Optional evaluation window start date for paper trading (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--eval-end",
        help="Optional evaluation window end date for paper trading (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["SPY", "BTC-USD"],
        help="Benchmarks used during the paper test evaluation window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, frontier_df, alloc_df = run_portfolio_example(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        ipopt_path=args.ipopt_path,
        n_points=args.n_points,
        output_dir=args.output_dir,
        show_plots=not args.no_show,
    )

    if args.eval_start and args.eval_end:
        weights = select_max_return_weights(frontier_df, alloc_df)
        evaluation = evaluate_out_of_sample(
            tickers=args.tickers,
            weights=weights,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            benchmark_tickers=args.benchmarks,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = output_dir / "paper_test_results.csv"
        evaluation.to_csv(evaluation_path, index=False)
        print("Paper trading evaluation saved to", evaluation_path)
        print(evaluation)


if __name__ == "__main__":
    main()
