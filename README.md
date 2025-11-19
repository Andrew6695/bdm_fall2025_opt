# Mean-Variance Portfolio Optimization Pipeline

This repository packages the original working notebook into a clean GitHub-style
project. It downloads historical prices from Yahoo Finance, converts them to
monthly returns, sweeps a variance cap to trace the efficient frontier, and
visualizes both the frontier and the corresponding allocations. Every run saves
publication-ready figures and evaluation tables inside `output/` for easy
grading and documentation.

## Quick Start
1. **Clone** the repository and enter the folder:
   ```bash
   git clone https://github.com/<your-username>/bdm_fall2025_opt.git
   cd bdm_fall2025_opt
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\\Scripts\\Activate.ps1  # Windows PowerShell
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the CLI (with explicit training window):**
   ```bash
   python main.py --tickers GE KO NVDA --start-date 2020-01-01 --end-date 2024-01-01
   ```

## Running in Google Colab
1. Upload (or clone) the repo inside your Colab workspace.
2. Install the scientific stack plus IPOPT extensions:
   ```python
   !pip install -r requirements.txt
   !idaes get-extensions --to /content/bin
   %matplotlib inline
   import matplotlib
   matplotlib.use('module://matplotlib_inline.backend_inline')
   ```
3. Point the CLI to the freshly installed IPOPT binary:
   ```python
   !python main.py --tickers GE KO NVDA --ipopt-path /content/bin/ipopt
   ```
4. The script will display two matplotlib figures in the output cell: the
   efficient frontier and the allocation-by-risk chart. The exact PNGs are
   automatically archived under `/content/bdm_fall2025_opt/output/` so they can
   be downloaded as supporting evidence.

## Running on macOS (local machine)
1. Make sure Homebrew has installed a recent Python (3.10+ recommended).
2. From the project root, create/activate a virtual environment and install the
   requirements (see Quick Start). IPOPT is pulled in through `idaes-pse`, so no
   manual compilation is necessary.
3. Execute the CLI. For example (training data spans 2018–2024 while plots and
   CSVs are written to `output/`):
   ```bash
   python main.py --tickers AAPL MSFT NVDA --start-date 2018-01-01 --end-date 2024-01-01
   ```

## CLI Usage
The CLI mirrors the professor's preferred format and exposes the core knobs:
```bash
python main.py \
  --tickers GE KO NVDA \
  --start-date 2020-01-01 \
  --end-date 2024-01-01 \
  --n-points 250 \
  --ipopt-path /content/bin/ipopt \
  --output-dir output \
  --eval-start 2024-08-01 --eval-end 2024-10-31 \
  --benchmarks SPY BTC-USD
```
- `--tickers`: space-separated list of equities.
- `--start-date` / `--end-date`: inclusive date range pulled from Yahoo Finance.
- `--n-points`: number of variance caps (higher = denser frontier).
- `--ipopt-path`: location of the IPOPT solver installed via `idaes get-extensions`.
- `--output-dir`: folder that stores `efficient_frontier.png`,
  `allocation_vs_risk.png`, and optional evaluation CSVs.
- `--no-show`: skip GUI pop-ups (useful on headless servers/CI).
- `--eval-start` / `--eval-end`: enables paper testing on a hold-out window.
- `--benchmarks`: benchmarks to compare against during the paper test (defaults
  to S&P 500 via `SPY` and Bitcoin via `BTC-USD`).

### Paper testing example (Jan 2024 – Oct 2025)
Use a rolling split to match the professor's specification—train on January 2024
through July 2025, then evaluate on August, September, and October 2025:
```bash
python main.py \
  --tickers GE KO NVDA \
  --start-date 2024-01-01 --end-date 2025-07-31 \
  --eval-start 2025-08-01 --eval-end 2025-10-31 \
  --benchmarks SPY BTC-USD \
  --output-dir output \
  --n-points 250
```
The script will:
1. Train on the January 2024 – July 2025 data and trace the frontier.
2. Select the maximum-return allocation.
3. Evaluate that allocation on August–October 2025 and compare it against
   (a) equal weighting, (b) buying the S&P 500 (`SPY`), and (c) buying Bitcoin
   (`BTC-USD`).
4. Save the full comparison table to `output/paper_test_results.csv` so you can
   cite out-of-sample performance in the write-up.

## Example Output
Running the example command produces two figures (both written to `output/`):
1. **Efficient Frontier:** risk (variance) on the x-axis, expected monthly return
   on the y-axis. The dots correspond to IPOPT solutions under progressively
   looser variance caps.
2. **Allocation vs Risk:** each line shows how a ticker's weight changes as the
   risk budget grows. The plot confirms that allocations remain in the
   long-only, fully-invested region (0–1 on the y-axis).
3. **(Optional) Paper-test table:** if you specified evaluation dates, the CSV
   in `output/` reports cumulative and average monthly returns for the optimized
   weights, an equal-weight baseline, S&P 500, and Bitcoin so you can compare
   strategies at a glance.

## Project Layout
```
bdm_fall2025_opt/
├── main.py                  # CLI entry point
├── src/portfolio_pipeline.py# Data download, Pyomo model, plotting
├── README.md                # You're reading it
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Notes on IPOPT via IDAES
- `idaes-pse` ships a helper (`idaes get-extensions`) that downloads IPOPT and
  compiles it for the current platform.
- In Colab, the solver binary typically lands under `/content/bin/ipopt`; on a
  local machine it will be inside `~/.idaes/bin/ipopt` unless you override the
  target folder.
- Pass the exact path to `--ipopt-path` if it differs from the default defined
  in `src/portfolio_pipeline.py`.

Happy optimizing!
