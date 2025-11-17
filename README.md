# Mean-Variance Portfolio Optimization Pipeline

This repository downloads historical prices from Yahoo Finance, converts them to
monthly returns, sweeps a variance cap to trace the efficient frontier, and
visualizes both the frontier and the corresponding allocations.

## Quick Start
1. **Clone** the repository and enter the folder:
   ```bash
   git clone https://github.com/<your-username>/bdm_fall2025_opt.git
   cd bdm_fall2025_opt
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\\Scripts\\Activate.ps1  # Windows PowerShell
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the CLI:**
   ```bash
   python main.py --tickers GE KO NVDA --start-date 2020-01-01 --end-date 2024-01-01
   ```

## Running in Google Colab
1. Upload (or clone) the repo inside your Colab workspace.
2. Install the scientific stack plus IPOPT extensions:
   ```python
   !pip install -r requirements.txt
   !idaes get-extensions --to /content/bin
   ```
3. Point the CLI to the freshly installed IPOPT binary:
   ```python
   !python main.py --tickers GE KO NVDA --ipopt-path /content/bin/ipopt
   ```
4. The script will display two matplotlib figures in the output cell: the
   efficient frontier and the allocation-by-risk chart.

## Running on macOS local machine
1. Make sure Homebrew has installed a recent Python (3.10+ recommended).
2. From the project root, create/activate a virtual environment and install the
   requirements (see Quick Start). IPOPT is pulled in through `idaes-pse`, so no
   manual compilation is necessary.
3. Execute the CLI. For example:
   ```bash
   python main.py --tickers AAPL MSFT NVDA --start-date 2018-01-01 --end-date 2024-01-01
   ```

## CLI Usage
The CLI exposes the core knobs:
```bash
python main.py \
  --tickers GE KO NVDA \
  --start-date 2020-01-01 \
  --end-date 2024-01-01 \
  --n-points 250 \
  --ipopt-path /content/bin/ipopt
```
- `--tickers`: space-separated list of equities.
- `--start-date` / `--end-date`: inclusive date range pulled from Yahoo Finance.
- `--n-points`: number of variance caps (higher = denser frontier).
- `--ipopt-path`: location of the IPOPT solver installed via `idaes get-extensions`.

## Example Output
Running the example command produces two figures:
1. **Efficient Frontier:** risk (variance) on the x-axis, expected monthly return
   on the y-axis. The dots correspond to IPOPT solutions under progressively
   looser variance caps.
2. **Allocation vs Risk:** Each line shows how a ticker's weight changes as the
   risk budget grows. The plot confirms that allocations remain in the
   long-only, fully-invested region (0–1 on the y-axis).

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
  local machine, it will be inside `~/.idaes/bin/ipopt` unless you override the
  target folder.
- Pass the exact path to `--ipopt-path` if it differs from the default defined
  in `src/portfolio_pipeline.py`.
