"""
Ambiguity Aversion Analysis and Visualization

This script analyzes results from ambiguity preference experiments and fits
sigmoid models to understand AI behavior under ambiguity vs. known risk.

WHAT THIS SCRIPT DOES:
- Loads ambiguity game experimental results from JSON files
- Fits sigmoid utility models using maximum likelihood estimation
- Generates publication-quality plots and statistical summaries
- Creates comprehensive CSV and PNG outputs for analysis
- Handles both neutral and persona-intervention experiments

THEORETICAL MODEL:
The analysis fits a sigmoid choice model where:
- ε (epsilon): Degree of ambiguity aversion (0=complete aversion, 1=no aversion)
- β (beta): Sensitivity to utility differences
- R*: Critical red ball count where preference switches

MODEL FORMULATION:
P(choose ambiguous) = 1 / (1 + exp(-β * (VA - VR)))
Where VA = (1-ε) * 0.5, VR = R/100 (R = red balls in risky urn)

REQUIREMENTS:
- Python 3.8+
- Required packages: numpy, pandas, matplotlib, scipy, argparse
- Ambiguity game result files in JSON format

USAGE:
1. Run from ambiguity-games/data-analyze/ directory
2. Default: python analyze_ambiguity_results.py (searches results/ recursively)
3. Custom: python analyze_ambiguity_results.py --root /path/to/results --pattern "*.json"

OUTPUT:
- Per-model CSV files with detailed statistics
- Individual model fit plots (PNG)
- Summary CSV with all model comparisons
- Summary table (PNG) for publications
- All outputs saved to organized directory structure
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

# ============= CONFIGURATION =============
# Modify these values to customize the analysis

# Directory paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"  # Where result files are located
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "ambiguity-games" / "analysis" / "results"
)  # Where outputs are saved
# TODO: Set your input/output directories here if different from defaults

# Analysis parameters
DEFAULT_FILE_PATTERN = "ambiguity_results_*.json"  # Pattern to match result files
RIDGE_PENALTY = 1e-6  # Regularization parameter for optimization
PLOT_DPI = 150  # DPI for plot outputs
TABLE_DPI = 200  # DPI for table outputs

# Optimization settings
OPTIMIZATION_METHOD = "L-BFGS-B"
INITIAL_GUESS = np.array([0.0, 1.0])  # Initial [eps_logit, beta] values

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============= END CONFIGURATION =============

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Create output directory
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _relpath(p: Path) -> str:
    """
    Convert path to relative string representation for logging.

    Args:
        p: Path to convert

    Returns:
        Relative path string with forward slashes
    """
    try:
        return str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


@dataclass
class FitResult:
    """
    Container for sigmoid model fitting results.

    Stores all parameters and metadata from fitting a sigmoid choice model
    to ambiguity preference experimental data.
    """

    model: str  # AI model name
    experiment_type: str  # "neutral" or "opportunity_hunter"
    persona_used: bool  # Whether persona intervention was used
    epsilon: float  # Ambiguity aversion parameter (0-1)
    beta: float  # Choice sensitivity parameter
    r_star: float  # Critical red ball count where preference switches
    amb_pct: float  # Overall proportion choosing ambiguous (0-1)
    risk_pct: float  # Overall proportion choosing risk (0-1)
    n_choices: int  # Total number of choices made
    source_file: Path  # Path to source data file
    method: str  # Fitting method used


# ---------------- Core mathematical functions ----------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid (logistic) function.

    Args:
        z: Input values

    Returns:
        Sigmoid-transformed values in (0,1) range
    """
    return 1.0 / (1.0 + np.exp(-z))


def loglik(
    theta: np.ndarray,
    xs: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    ridge: float = RIDGE_PENALTY,
) -> float:
    """
    Compute negative log-likelihood for sigmoid choice model with ridge penalty.

    The model assumes: P(choose ambiguous) = sigmoid(β * (VA - VR))
    where VA = (1-ε) * 0.5 and VR = R/100

    Args:
        theta: Parameter vector [eps_logit, beta]
        xs: Red ball counts (independent variable)
        successes: Number of ambiguous choices per condition
        trials: Total number of trials per condition
        ridge: Ridge penalty coefficient for regularization

    Returns:
        Negative log-likelihood plus ridge penalty
    """
    eps = 1 / (1 + np.exp(-theta[0]))  # Logistic transform: keep eps in [0,1]
    beta = theta[1]

    # Compute utilities
    VA = (1 - eps) * 0.5  # Utility of ambiguous option
    VB = xs / 100.0  # Utility of risky option (R/100)
    z = beta * (VA - VB)

    # Predicted probabilities with numerical stability
    p = sigmoid(z)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    # Log-likelihood
    ll = (successes * np.log(p) + (trials - successes) * np.log(1 - p)).sum()

    # Add ridge penalty on beta for regularization
    return -ll + ridge * (beta**2)


def fit_eps_beta(
    xs: List[int], successes: List[int], trials: List[int]
) -> Tuple[float, float, bool, float]:
    """
    Fit epsilon (ambiguity aversion) and beta (sensitivity) parameters.

    Uses maximum likelihood estimation with L-BFGS-B optimization
    to fit the sigmoid choice model to experimental data.

    Args:
        xs: Red ball counts for each condition
        successes: Number of ambiguous choices per condition
        trials: Total trials per condition

    Returns:
        Tuple of (epsilon, beta, success, final_nll):
        - epsilon: Fitted ambiguity aversion parameter (0-1)
        - beta: Fitted choice sensitivity parameter
        - success: Whether optimization converged
        - final_nll: Final negative log-likelihood value
    """
    # Convert inputs to numpy arrays
    xs, successes, trials = map(np.asarray, (xs, successes, trials))

    # Optimize parameters
    res = optimize.minimize(
        fun=lambda th: loglik(th, xs, successes, trials),
        x0=INITIAL_GUESS,
        method=OPTIMIZATION_METHOD,
    )

    # Transform fitted parameters
    eps = 1 / (1 + np.exp(-res.x[0]))  # Convert from logit to probability
    beta = res.x[1]

    # Calculate final NLL for diagnostics
    final_nll = loglik(res.x, xs, successes, trials)

    return eps, beta, res.success, final_nll


# ---------------- Input/Output helper functions ----------------


def parse_result_file(
    fp: Path,
) -> Tuple[Dict[int, Dict[str, int]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Parse ambiguity game result file into structured data.

    Extracts choice counts by red ball condition, metadata, and raw records
    from the JSON result file format.

    Args:
        fp: Path to the JSON result file

    Returns:
        Tuple of (breakdown, meta, records):
        - breakdown: Dict mapping red ball count to choice counts
        - meta: Dict with model metadata (model, experiment_type, persona_used)
        - records: List of individual choice records
    """
    data = json.loads(fp.read_text(encoding="utf-8"))
    breakdown: Dict[int, Dict[str, int]] = {}

    # Extract choice counts by red ball condition
    pct_data = data.get("percentage_breakdown", {})
    for k, v in pct_data.items():
        try:
            red = int(k)
        except Exception:
            try:
                red = int(v.get("red_balls", k))
            except Exception:
                continue  # Skip malformed entries

        breakdown[red] = {
            "ambiguous_count": int(v.get("ambiguous_count", 0)),
            "risk_count": int(v.get("risk_count", 0)),
        }

    # Extract metadata
    meta = {
        "model": data.get("model", "unknown"),
        "experiment_type": data.get("experiment_type", "neutral"),
        "persona_used": bool(data.get("persona_used", False)),
    }

    records = data.get("all_records", [])
    return breakdown, meta, records


def compute_choice_shares(records: List[Dict[str, Any]]) -> Tuple[float, float, int]:
    """
    Compute overall choice shares from individual records.

    Args:
        records: List of individual choice records

    Returns:
        Tuple of (ambiguous_share, risk_share, total_count):
        - ambiguous_share: Proportion choosing ambiguous (0-1)
        - risk_share: Proportion choosing risk (0-1)
        - total_count: Total number of valid records
    """
    n = len(records)
    if n == 0:
        return 0.0, 0.0, 0

    ambiguous_count = sum(
        1 for r in records if str(r.get("choice", "")).lower() == "ambiguous"
    )
    risk_count = sum(1 for r in records if str(r.get("choice", "")).lower() == "risk")

    return ambiguous_count / n, risk_count / n, n


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """
    Write data rows to CSV file with specified fieldnames.

    Args:
        path: Output file path
        rows: List of dictionaries containing row data
        fieldnames: List of column headers
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------- Plotting and visualization functions ----------------


def plot_fits(
    per_red: List[Dict[str, Any]],
    eps: float,
    beta: float,
    model: str,
    exp: str,
    outfile: Path,
) -> None:
    """
    Create individual model fit plot comparing observed vs. predicted choices.

    Generates a scatter plot showing observed choice proportions vs. the fitted
    sigmoid model predictions for a specific model and experiment condition.

    Args:
        per_red: List of dictionaries with per-condition statistics
        eps: Fitted ambiguity aversion parameter
        beta: Fitted choice sensitivity parameter
        model: Model name for plot title
        exp: Experiment type for plot title
        outfile: Path to save the plot
    """
    xs = np.array([row["red_balls"] for row in per_red])
    obs = np.array([row["ambiguous_pct"] for row in per_red])

    # Compute fitted probabilities
    VA = (1 - eps) * 0.5  # Utility of ambiguous option
    fitted = sigmoid(beta * (VA - xs / 100))

    # Create plot
    plt.figure()
    plt.scatter(xs, obs, label="Observed", color="black", alpha=0.7, s=50)
    plt.plot(xs, fitted, label="Fitted", linewidth=2, color="red")
    plt.xlabel("Red balls in risky urn (R)")
    plt.ylabel("P(choose ambiguous)")
    plt.title(f"{model} ({exp}) sigmoid fit")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"📊 Saved fit plot: {_relpath(outfile)}")


def render_table_png(
    df: pd.DataFrame, outfile: Path, title: str = "Ambiguity Game – Summary"
) -> None:
    """
    Render pandas DataFrame as publication-quality PNG table.

    Args:
        df: DataFrame to render as table
        outfile: Path to save the table image
        title: Title for the table
    """
    # Calculate figure size based on content
    n_cols = len(df.columns)
    n_rows = len(df)
    fig_width = max(10, 0.22 * (n_cols + 2) * n_rows / 6)
    fig_height = max(3.5, 0.6 * n_rows)

    plt.figure(figsize=(fig_width, fig_height))
    plt.axis("off")

    # Create table
    table = plt.table(
        cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    plt.title(title, fontsize=12, pad=12)

    # Save table
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight", dpi=TABLE_DPI)
    plt.close()

    logger.info(f"🖼️  Saved table PNG: {_relpath(outfile)}")


# ---------------- Data processing functions ----------------


def process_file(fp: Path) -> Tuple[Optional[FitResult], List[Dict[str, Any]]]:
    """
    Process a single ambiguity game result file.

    Loads the file, fits the sigmoid model, and returns fitted parameters
    along with per-condition statistics.

    Args:
        fp: Path to the result file to process

    Returns:
        Tuple of (fit_result, per_red_data):
        - fit_result: FitResult object with fitted parameters, or None if failed
        - per_red_data: List of per-condition statistics dictionaries
    """
    # Parse file data
    breakdown, meta, records = parse_result_file(fp)
    xs = sorted(breakdown.keys())
    successes = [breakdown[x]["ambiguous_count"] for x in xs]
    trials = [breakdown[x]["ambiguous_count"] + breakdown[x]["risk_count"] for x in xs]

    # Diagnostic logging
    total_trials = sum(trials)
    logger.info(f"🔍 Processing: {_relpath(fp)}")
    logger.info(f"   Red ball levels tested: {xs}")
    logger.info(f"   Total trials: {total_trials}")

    # Check for complete separation (edge cases)
    for x, succ, n in zip(xs, successes, trials):
        if succ == 0:
            logger.warning(f"   Complete separation at red={x} (0 ambiguous choices)")
        elif succ == n:
            logger.warning(f"   Complete separation at red={x} (all ambiguous choices)")

    # Validate data
    if not xs or total_trials == 0:
        logger.warning("   SKIPPED: No valid data found")
        return None, []

    # Fit sigmoid model
    eps, beta, success, final_nll = fit_eps_beta(xs, successes, trials)
    logger.info(f"   Model fit success: {success}, Final NLL: {final_nll:.4f}")

    # Compute derived quantities
    r_star = 50 * (1 - eps)  # Critical red ball count
    amb_share, risk_share, n_choices = compute_choice_shares(records)

    # Generate per-condition predictions
    per_red = []
    VA = (1 - eps) * 0.5  # Utility of ambiguous option

    for x, succ, n in zip(xs, successes, trials):
        p_hat = sigmoid(beta * (VA - x / 100))  # Predicted probability
        per_red.append(
            {
                "red_balls": x,
                "ambiguous_count": succ,
                "risk_count": n - succ,
                "total": n,
                "ambiguous_pct": succ / n if n else 0,
                "p_hat": p_hat,
            }
        )

    # Create result object
    fit = FitResult(
        model=meta["model"],
        experiment_type=meta["experiment_type"],
        persona_used=meta["persona_used"],
        epsilon=eps,
        beta=beta,
        r_star=r_star,
        amb_pct=amb_share,
        risk_pct=risk_share,
        n_choices=n_choices,
        source_file=fp,
        method="scipy-mle",
    )

    return fit, per_red


# ---------------- Main analysis function ----------------


def main() -> None:
    """
    Main analysis function for ambiguity game results.

    Orchestrates the complete analysis pipeline:
    1. Discovers result files using recursive search
    2. Processes each file and fits sigmoid models
    3. Generates per-model outputs (CSV, plots)
    4. Creates summary statistics and visualizations
    5. Saves all outputs to organized directory structure
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze ambiguity game results and fit sigmoid models"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory to search for result files",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_FILE_PATTERN,
        help="File pattern to match result files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory to save analysis outputs (default: auto-generated)",
    )
    args = parser.parse_args()

    # Handle output directory
    global DEFAULT_OUTPUT_DIR
    if args.output is not None:
        DEFAULT_OUTPUT_DIR = args.output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("🎯 Starting Ambiguity Game Analysis")
    logger.info("=" * 50)
    logger.info(f"📂 Search root: {args.root}")
    logger.info(f"🔍 File pattern: {args.pattern}")
    logger.info(f"📁 Output directory: {DEFAULT_OUTPUT_DIR}")

    # File discovery with recursive search
    files = sorted(args.root.rglob(args.pattern))

    if not files:
        logger.error(f"No files found matching pattern '{args.pattern}' in {args.root}")
        return

    logger.info(f"📋 Found {len(files)} result files")
    if len(files) <= 10:
        for i, fp in enumerate(files, 1):
            logger.info(f"   {i}: {_relpath(fp)}")
    else:
        for i, fp in enumerate(files[:5], 1):
            logger.info(f"   {i}: {_relpath(fp)}")
        logger.info(f"   ... and {len(files) - 5} more files")
    logger.info("")

    # Initialize data structures
    summary_rows: List[Dict[str, Any]] = []
    skipped_files: List[Path] = []

    # Process each file
    for fp in files:
        fit, per_red = process_file(fp)
        if fit is None:
            skipped_files.append(fp)
            continue

        # Generate per-model outputs
        per_model_dir = DEFAULT_OUTPUT_DIR / "per_model"
        per_model_dir.mkdir(parents=True, exist_ok=True)

        # Per-model CSV with detailed statistics
        per_model_csv = per_model_dir / f"per_red_{fit.experiment_type}_{fit.model}.csv"
        write_csv(
            per_model_csv,
            per_red,
            [
                "red_balls",
                "ambiguous_count",
                "risk_count",
                "total",
                "ambiguous_pct",
                "p_hat",
            ],
        )

        # Per-model fit plot
        plot_file = per_model_dir / f"fitplot_{fit.experiment_type}_{fit.model}.png"
        plot_fits(
            per_red, fit.epsilon, fit.beta, fit.model, fit.experiment_type, plot_file
        )

        # Log results for this model
        logger.info(
            f"  ✅ {fit.model} ({fit.experiment_type}): ε={fit.epsilon:.4f}, β={fit.beta:.4f}, R*={fit.r_star:.3f}"
        )
        logger.info(f"     📊 CSV: {_relpath(per_model_csv)}")
        logger.info(f"     📈 PNG: {_relpath(plot_file)}")
        logger.info("")

        # Add to summary data
        summary_rows.append(
            {
                "model": fit.model,
                "experiment": fit.experiment_type,
                "persona": "yes" if fit.persona_used else "no",
                "epsilon": round(fit.epsilon, 4),
                "beta": round(fit.beta, 4),
                "R_star": round(fit.r_star, 3),
                "%ambiguous": round(100 * fit.amb_pct, 1),
                "%risk": round(100 * fit.risk_pct, 1),
                "n_choices": fit.n_choices,
                "method": fit.method,
                "source_file": _relpath(fp),
            }
        )

    # Validation and summary statistics
    processed_files = len(files) - len(skipped_files)
    if len(summary_rows) != processed_files:
        logger.warning(
            f"Data inconsistency: summary_rows={len(summary_rows)} != processed_files={processed_files}"
        )
        if skipped_files:
            logger.warning(f"Skipped files: {[_relpath(f) for f in skipped_files]}")

    # Generate summary outputs
    if not summary_rows:
        logger.error("No valid data found. Cannot generate summary outputs.")
        return

    # Sort summary by experiment type and model
    summary_rows.sort(key=lambda r: (r["experiment"], r["model"]))

    # Summary CSV
    summary_csv = DEFAULT_OUTPUT_DIR / "sigmoid_fit_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "model",
            "experiment",
            "persona",
            "epsilon",
            "beta",
            "R_star",
            "%ambiguous",
            "%risk",
            "n_choices",
            "method",
            "source_file",
        ],
    )
    logger.info(f"💾 Saved summary CSV: {_relpath(summary_csv)}")

    # Summary statistics
    models = set(r["model"] for r in summary_rows)
    experiments = set(r["experiment"] for r in summary_rows)
    logger.info(f"📊 Summary statistics:")
    logger.info(f"   Total models processed: {len(summary_rows)}")
    logger.info(f"   Distinct models: {len(models)} ({sorted(models)})")
    logger.info(f"   Distinct experiments: {len(experiments)} ({sorted(experiments)})")

    # Verify output files exist
    check_count = min(3, len(summary_rows))
    logger.info("🔍 Verifying output files:")
    for i in range(check_count):
        row = summary_rows[i]
        csv_path = (
            DEFAULT_OUTPUT_DIR
            / "per_model"
            / f"per_red_{row['experiment']}_{row['model']}.csv"
        )
        png_path = (
            DEFAULT_OUTPUT_DIR
            / "per_model"
            / f"fitplot_{row['experiment']}_{row['model']}.png"
        )
        logger.info(
            f"   File {i+1}: CSV exists={csv_path.exists()}, PNG exists={png_path.exists()}"
        )

    # Summary table PNG
    df = pd.DataFrame(summary_rows)
    fmt_df = df.copy()

    # Format numeric columns for display
    for col in ["epsilon", "beta", "R_star"]:
        fmt_df[col] = fmt_df[col].map(lambda v: f"{v:.3f}")
    for col in ["%ambiguous", "%risk"]:
        fmt_df[col] = fmt_df[col].map(lambda v: f"{v:.1f}")
    fmt_df["n_choices"] = fmt_df["n_choices"].astype(str)

    table_png = DEFAULT_OUTPUT_DIR / "sigmoid_fit_summary.png"
    render_table_png(fmt_df, table_png, title="Ambiguity Game Sigmoid Model Summary")
    logger.info(f"🖼️  Saved summary table: {_relpath(table_png)}")

    logger.info("\n✅ Analysis complete! All outputs saved to:")
    logger.info(f"   📁 {DEFAULT_OUTPUT_DIR}")
    logger.info("   📊 Ready for publication and further analysis")


if __name__ == "__main__":
    main()
