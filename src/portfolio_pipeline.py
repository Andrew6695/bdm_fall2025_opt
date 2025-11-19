"""Portfolio optimization pipeline module.

This module exposes helper functions for downloading data, building and solving
mean-variance models, and visualizing results. All functions are direct
translations of the working notebook logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

try:  # pragma: no cover - optional inline backend for notebooks
    from IPython import get_ipython

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic("matplotlib", "inline")
except Exception:
    pass

try:  # pragma: no cover - ensure inline backend in Colab/Jupyter
    import matplotlib

    matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass

from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    NonNegativeReals,
    Param,
    Objective,
    maximize,
    Constraint,
)
from pyomo.opt import SolverFactory, TerminationCondition


IPOPT_PATH = "/content/bin/ipopt"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_monthly_returns(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance and convert to monthly returns."""

    price_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                print(f"[warn] no data for {ticker}, skipping")
                continue
            if "Close" not in df.columns:
                print(f"[warn] 'Close' column missing for {ticker}, skipping")
                continue

            close_series = df["Close"]
            if not close_series.empty and isinstance(close_series.index, pd.DatetimeIndex):
                price_dict[ticker] = close_series
            else:
                print(
                    f"[warn] Skipping {ticker} due to empty or malformed 'Close' series/index"
                )

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[error] downloading {ticker}: {exc}")

    if not price_dict:
        raise RuntimeError("No valid price data downloaded for any ticker. Check tickers/date range.")

    daily_prices = pd.concat(price_dict.values(), axis=1, keys=price_dict.keys())
    daily_prices = daily_prices.dropna(how="all")
    daily_returns = daily_prices.pct_change().dropna(how="all")

    monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
    monthly_returns = monthly_returns.dropna(how="any")

    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    print("Monthly returns shape:", monthly_returns.shape)
    return monthly_returns


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------

def build_markowitz_model(returns_df: pd.DataFrame):
    """Build a Pyomo Markowitz model (long-only, fully invested)."""

    assets = list(returns_df.columns)
    mu = returns_df.mean()
    sigma = returns_df.cov()

    model = ConcreteModel()
    model.Assets = Set(initialize=assets)
    model.x = Var(model.Assets, within=NonNegativeReals, bounds=(0, 1))
    model.mu = Param(model.Assets, initialize=mu.to_dict())

    sigma_dict = {(i, j): float(sigma.loc[i, j]) for i in assets for j in assets}
    model.Sigma = Param(model.Assets, model.Assets, initialize=sigma_dict)

    def total_return(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)

    model.obj = Objective(rule=total_return, sense=maximize)

    def budget(m):
        return sum(m.x[a] for a in m.Assets) == 1

    model.budget = Constraint(rule=budget)

    return model, assets, mu, sigma


def portfolio_variance(weights: Sequence[float], sigma: pd.DataFrame | np.ndarray) -> float:
    weights_arr = np.array(weights, dtype=float)
    sigma_arr = sigma.values if isinstance(sigma, pd.DataFrame) else sigma
    return float(weights_arr @ sigma_arr @ weights_arr)


def sweep_efficient_frontier(
    returns_df: pd.DataFrame,
    ipopt_path: str = IPOPT_PATH,
    n_points: int = 200,
):
    """Solve the efficient frontier by sweeping variance caps."""

    model, assets, mu, sigma = build_markowitz_model(returns_df)
    sigma_np = sigma.values
    n_assets = len(assets)

    eq_weights = np.ones(n_assets) / n_assets
    min_var = portfolio_variance(eq_weights, sigma_np)
    max_var_single = float(np.max(np.diag(sigma_np)))

    min_cap = max(min_var * 0.5, 1e-8)
    max_cap = max(max_var_single * 1.5, min_cap * 5)
    caps = np.linspace(min_cap, max_cap, n_points)

    solver = SolverFactory("ipopt", executable=ipopt_path)

    frontier_data = {"Risk": [], "Return": []}
    alloc_data = {asset: [] for asset in assets}
    alloc_data["Risk"] = []

    print(
        f"Solving {len(caps)} portfolio problems from cap={min_cap:.3e} to {max_cap:.3e}..."
    )

    for cap in caps:
        if hasattr(model, "risk_constraint"):
            model.del_component(model.risk_constraint)

        def risk_con(m):
            return (
                sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets)
                <= cap
            )

        model.risk_constraint = Constraint(rule=risk_con)

        result = solver.solve(model, tee=False)
        term = result.solver.termination_condition

        if term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
            continue

        weights = [model.x[a]() for a in assets]
        realized_var = portfolio_variance(weights, sigma_np)
        realized_ret = float(np.dot(mu.values, np.array(weights)))

        frontier_data["Risk"].append(realized_var)
        frontier_data["Return"].append(realized_ret)

        alloc_data["Risk"].append(realized_var)
        for asset, weight in zip(assets, weights):
            alloc_data[asset].append(weight)

    if len(frontier_data["Risk"]) == 0:
        raise RuntimeError("No feasible portfolios found. Try different tickers/dates.")

    frontier_df = pd.DataFrame(frontier_data).sort_values("Risk").reset_index(drop=True)
    alloc_df = pd.DataFrame(alloc_data).sort_values("Risk").set_index("Risk")

    return frontier_df, alloc_df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_frontier(
    frontier_df: pd.DataFrame,
    output_dir: str | Path = "output",
    show: bool = True,
) -> Path:
    """Plot the efficient frontier and save it under ``output_dir``."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frontier_df["Risk"], frontier_df["Return"], marker="o", linestyle="-", markersize=3)
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Portfolio Risk (Variance)")
    ax.set_ylabel("Expected Monthly Return")
    ax.grid(True)
    fig.tight_layout()

    figure_path = output_path / "efficient_frontier.png"
    fig.savefig(figure_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return figure_path


def plot_allocations(
    alloc_df: pd.DataFrame,
    output_dir: str | Path = "output",
    show: bool = True,
) -> Path:
    """Plot allocation weights as a function of portfolio risk and save to ``output_dir``."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in alloc_df.columns:
        ax.plot(
def plot_frontier(frontier_df: pd.DataFrame) -> None:
    """Plot the efficient frontier."""

    plt.figure(figsize=(8, 5))
    plt.plot(frontier_df["Risk"], frontier_df["Return"], marker="o", linestyle="-", markersize=3)
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Monthly Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_allocations(alloc_df: pd.DataFrame) -> None:
    """Plot allocation weights as a function of portfolio risk."""

    plt.figure(figsize=(10, 6))
    for col in alloc_df.columns:
        plt.plot(
            alloc_df.index,
            alloc_df[col],
            marker="o",
            markersize=3,
            linewidth=0.7,
            label=str(col),
        )
    ax.set_title("Optimal Allocation vs Portfolio Risk")
    ax.set_xlabel("Portfolio Risk (Variance)")
    ax.set_ylabel("Proportion Invested")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    figure_path = output_path / "allocation_vs_risk.png"
    fig.savefig(figure_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return figure_path
    plt.title("Optimal Allocation vs Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_portfolio_example(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    ipopt_path: str = IPOPT_PATH,
    n_points: int = 200,
    output_dir: str | Path = "output",
    show_plots: bool = True,
):
    """Download data, solve the frontier, and plot results."""

    monthly_returns = download_monthly_returns(tickers, start_date, end_date)
    print("Monthly returns head:")
    print(monthly_returns.head())

    frontier_df, alloc_df = sweep_efficient_frontier(
        monthly_returns,
        ipopt_path=ipopt_path,
        n_points=n_points,
    )

    plot_frontier(frontier_df, output_dir=output_dir, show=show_plots)
    plot_allocations(alloc_df, output_dir=output_dir, show=show_plots)

    return monthly_returns, frontier_df, alloc_df


def select_max_return_weights(
    frontier_df: pd.DataFrame, alloc_df: pd.DataFrame
) -> pd.Series:
    """Return the weight vector associated with the maximum-return frontier point."""

    max_return_idx = frontier_df["Return"].idxmax()
    target_risk = frontier_df.loc[max_return_idx, "Risk"]
    weights = alloc_df.loc[target_risk]
    return weights


def evaluate_out_of_sample(
    tickers: Sequence[str],
    weights: pd.Series,
    eval_start: str,
    eval_end: str,
    benchmark_tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Evaluate trained weights against benchmarks on an evaluation window."""

    eval_returns = download_monthly_returns(tickers, eval_start, eval_end)
    weights_vector = weights.loc[eval_returns.columns]

    model_series = eval_returns @ weights_vector.values
    eq_weights = np.ones(len(eval_returns.columns)) / len(eval_returns.columns)
    equal_weight_series = eval_returns @ eq_weights

    evaluation_rows = [
        (
            "Optimized Portfolio",
            (1 + model_series).prod() - 1,
            model_series.mean(),
            model_series.shape[0],
        ),
        (
            "Equal Weight",
            (1 + equal_weight_series).prod() - 1,
            equal_weight_series.mean(),
            equal_weight_series.shape[0],
        ),
    ]

    if benchmark_tickers:
        for benchmark in benchmark_tickers:
            bench_returns = download_monthly_returns([benchmark], eval_start, eval_end)
            bench_series = bench_returns.iloc[:, 0]
            evaluation_rows.append(
                (
                    benchmark,
                    (1 + bench_series).prod() - 1,
                    bench_series.mean(),
                    bench_series.shape[0],
                )
            )

    eval_df = pd.DataFrame(
        evaluation_rows,
        columns=["Strategy", "CumulativeReturn", "AverageMonthlyReturn", "Months"],
    )
    return eval_df
    plot_frontier(frontier_df)
    plot_allocations(alloc_df)

    return monthly_returns, frontier_df, alloc_df
