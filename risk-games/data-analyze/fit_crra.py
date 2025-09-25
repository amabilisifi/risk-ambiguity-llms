#!/usr/bin/env python3
"""
CRRA (Constant Relative Risk Aversion) Analysis for AI Risk Preferences

This script analyzes AI language models' risk preferences using Constant Relative Risk Aversion
(CRRA) utility functions. It fits CRRA parameters to choice data from risk-reward experiments
and generates comprehensive visualizations and statistical analyses.

WHAT THIS SCRIPT DOES:
- Loads and processes risk game choice data from JSON files
- Fits CRRA utility function parameters using maximum likelihood estimation
- Generates individual and comparative plots of risk preferences
- Provides statistical summaries and risk preference interpretations

CRRA MODEL:
The CRRA utility function is: U(w) = (w^(1-r) - 1)/(1-r) for r ≠ 1, U(w) = ln(w) for r = 1
- r < 0: Risk seeking behavior
- r ≈ 0: Risk neutral behavior
- r > 0: Risk averse behavior (higher r = more risk averse)

DIFFERENCE FROM CARA:
Unlike CARA (which applies to payoffs), CRRA applies to total wealth levels,
making it more appropriate for modeling behavior over multiple decisions.

REQUIREMENTS:
- Python 3.8+
- Required packages: numpy, scipy, matplotlib, collections, json, pathlib
- Risk game experiment data in JSON format

USAGE:
1. Run from risk-games/data-analyze/ directory
2. Default: python fit_crra.py (analyzes available models)
3. Custom file: python fit_crra.py /path/to/results.json
4. Custom output: python fit_crra.py input.json output_dir

OUTPUT:
- JSON results files with fitted parameters and statistics
- PNG visualization plots (individual and combined)
- Comprehensive analysis summaries in console
"""

import collections
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# ============= CONFIGURATION =============
# Modify these values to customize the analysis

# Model configuration
AVAILABLE_MODELS = [
    "o3-mini-opportunity-hunter",
]  # Models to analyze by default

# Economic parameters
BACKGROUND_WEALTH = 100  # Initial wealth assumption (must match game setup)
# TODO: Set your background wealth assumption here if different

# File paths and directories
DEFAULT_INPUT_DIR = "../results"  # Relative to script location
DEFAULT_OUTPUT_DIR = "analysis"  # Relative to script location
# TODO: Set your input/output directories here if different from defaults

# Analysis parameters
RISK_AVERSION_BOUNDS = (-3.0, 5.0)  # Allowable range for r parameter
CHOICE_SENSITIVITY_BOUNDS = (0.01, 100.0)  # Allowable range for β parameter (> 0)

# Optimization settings
MAX_OPTIMIZATION_ITERATIONS = 3000
OPTIMIZATION_TOLERANCE = 1e-9
STARTING_POINTS = [
    (0.5, 1.0),  # Moderate risk aversion
    (1.0, 2.0),  # Log utility
    (2.0, 0.5),  # High risk aversion
    (0.1, 5.0),  # Low risk aversion
    (1.5, 10.0),  # High risk aversion, high beta
    (-0.5, 1.0),  # Risk seeking
]

# Plotting configuration
FIGURE_SIZE_INDIVIDUAL = (10, 7)
FIGURE_SIZE_COMBINED = (14, 9)
PLOT_DPI = 300
PLOT_STYLE = "default"

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============= END CONFIGURATION =============

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# Parse command line arguments
def parse_arguments() -> Tuple[Optional[List[str]], Optional[Path], Path]:
    """
    Parse command line arguments for the analysis script.

    Returns:
        Tuple of (models_to_process, input_path, output_directory)
        - models_to_process: List of model names or None for custom file
        - input_path: Path to custom input file or None for default models
        - output_directory: Directory to save results
    """
    if len(sys.argv) > 1 and sys.argv[1] in AVAILABLE_MODELS:
        models_to_process = [sys.argv[1]]
        input_path = None
    elif len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        models_to_process = None
    else:
        models_to_process = AVAILABLE_MODELS.copy()
        input_path = None

    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    return models_to_process, input_path, output_dir


# Parse arguments and setup
models_to_process, INPUT, OUTDIR = parse_arguments()
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

logger.info("🎯 CRRA Analysis for AI Risk Preferences")
logger.info("=" * 60)
logger.info(f"📊 Background wealth assumption: {BACKGROUND_WEALTH} tokens")
logger.info(f"📁 Output directory: {OUTDIR}")
logger.info(f"📅 Analysis timestamp: {STAMP}")


def crra_utility(wealth: float, r: float) -> float:
    """
    Compute Constant Relative Risk Aversion (CRRA) utility for a wealth level.

    The CRRA utility function is: U(w) = (w^(1-r) - 1)/(1-r) for r ≠ 1, U(w) = ln(w) for r = 1

    Args:
        wealth: The total wealth amount (must be positive)
        r: Risk aversion parameter
            - r < 0: Risk seeking behavior
            - r ≈ 0: Risk neutral behavior
            - r > 0: Risk averse behavior (higher r = more risk averse)
            - r = 1: Logarithmic utility (special case)

    Returns:
        Utility value for the given wealth level

    Raises:
        Returns -inf if wealth <= 0 (invalid wealth)
    """
    if wealth <= 0:
        return -np.inf

    if abs(r - 1) < 1e-8:  # r ≈ 1 (log utility)
        return np.log(wealth)
    else:
        return (wealth ** (1 - r) - 1) / (1 - r)


def expected_utility_safe(
    safe_payoff: float, background_wealth: float, r: float
) -> float:
    """Compute expected utility of the safe option."""
    return crra_utility(background_wealth + safe_payoff, r)


def expected_utility_risky(
    risky_payoff: float, prob_win: float, background_wealth: float, r: float
) -> float:
    """
    Compute expected utility of the risky option.

    The risky option adds risky_payoff to background wealth if successful,
    and leaves background wealth unchanged if unsuccessful.

    Args:
        risky_payoff: Payoff amount added to wealth if the risky option succeeds
        prob_win: Probability of success for the risky option
        background_wealth: Initial wealth level
        r: Risk aversion parameter

    Returns:
        Expected utility of the risky option
    """
    wealth_if_win = background_wealth + risky_payoff
    wealth_if_lose = background_wealth  # Win 0, keep background wealth

    eu_win = crra_utility(wealth_if_win, r)
    eu_lose = crra_utility(wealth_if_lose, r)

    return prob_win * eu_win + (1 - prob_win) * eu_lose


def choice_probability(eu_risky: float, eu_safe: float, beta: float) -> float:
    """
    Compute probability of choosing the risky option using logit model.

    Uses logistic regression where the probability depends on the
    difference in expected utilities between risky and safe options.

    Args:
        eu_risky: Expected utility of risky option
        eu_safe: Expected utility of safe option
        beta: Choice sensitivity parameter (higher β = more sensitive to differences)

    Returns:
        Probability of choosing the risky option (between 0 and 1)
    """
    utility_diff = eu_risky - eu_safe
    utility_diff = np.clip(utility_diff, -500, 500)  # Prevent overflow
    return expit(beta * utility_diff)


def negative_log_likelihood(
    params: Tuple[float, float], trials: List[Dict[str, Any]], background_wealth: float
) -> float:
    """
    Compute negative log-likelihood for CRRA parameter estimation.

    Uses maximum likelihood estimation to fit CRRA parameters to observed
    choice data. Penalizes invalid parameter combinations heavily.

    Args:
        params: Tuple of (r, beta) where r is risk aversion, beta is choice sensitivity
        trials: List of trial dictionaries with choice data
        background_wealth: Initial wealth assumption

    Returns:
        Negative log-likelihood value (lower is better fit)

    Note:
        Returns large penalty (1e10) for invalid parameters or numerical errors
    """
    r, beta = params

    # Parameter bounds check
    if beta <= 0:
        return 1e10

    nll = 0.0

    for trial in trials:
        try:
            # Calculate expected utilities
            eu_safe = expected_utility_safe(trial["safe_payoff"], background_wealth, r)
            eu_risky = expected_utility_risky(
                trial["risky_payoff"], trial["prob_win"], background_wealth, r
            )

            # Probability of choosing risky
            p_risky = choice_probability(eu_risky, eu_safe, beta)
            p_risky = np.clip(p_risky, 1e-12, 1 - 1e-12)  # Prevent log(0)

            # Add to log-likelihood
            m, n = trial["risky_choices"], trial["total_choices"]
            nll += -(m * np.log(p_risky) + (n - m) * np.log(1 - p_risky))

        except (OverflowError, ValueError, ZeroDivisionError) as e:
            logger.warning(
                f"Numerical error in trial {trial.get('id', 'unknown')}: {e}"
            )
            return 1e10

    return nll


def load_and_process_data(input_file: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Load and process risk game choice data from JSON file.

    Handles both old and new data formats, validates scenarios,
    and converts to trial format for analysis.

    Args:
        input_file: Path to JSON file containing choice data

    Returns:
        List of trial dictionaries, or None if loading/parsing fails

    Each trial dict contains:
        - id: Scenario identifier
        - prob_win: Win probability for risky option
        - safe_payoff: Guaranteed payoff for safe option
        - risky_payoff: Payoff if risky option succeeds
        - risky_choices: Number of times risky option was chosen
        - total_choices: Total number of choices in this scenario
        - risky_rate: Proportion of risky choices
    """
    try:
        with open(input_file, encoding="utf-8") as f:
            raw_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading {input_file}: {e}")
        return None

    # Count choices by scenario
    choice_counts = collections.Counter()
    scenario_info = {}

    for decision in raw_data:
        # Handle new data format
        scenario_id = decision.get("scenario_id", decision.get("id"))
        choice = decision.get("choice")

        # Skip invalid entries
        if not choice or choice not in ["safe", "risky"]:
            continue

        # Store scenario parameters (updated for new format)
        if scenario_id not in scenario_info:
            scenario_info[scenario_id] = {
                "prob_win": decision.get(
                    "probability", decision.get("p_red", decision.get("p"))
                ),
                "safe_payoff": decision.get("safe_amount", decision.get("constant")),
                "risky_payoff": decision.get("risky_reward"),
            }

        # Count choices
        choice_counts[(scenario_id, choice)] += 1

    # Convert to trial format
    trials = []
    for scenario_id, info in scenario_info.items():
        risky_choices = choice_counts.get((scenario_id, "risky"), 0)
        safe_choices = choice_counts.get((scenario_id, "safe"), 0)
        total_choices = risky_choices + safe_choices

        if total_choices == 0:
            continue

        # Validate scenario data
        if (
            any(v is None for v in info.values())
            or info["prob_win"] <= 0
            or info["prob_win"] >= 1
        ):
            logger.warning(f"Skipping invalid scenario {scenario_id}: {info}")
            continue

        trials.append(
            {
                "id": scenario_id,
                "prob_win": info["prob_win"],
                "safe_payoff": info["safe_payoff"],
                "risky_payoff": info["risky_payoff"],
                "risky_choices": risky_choices,
                "total_choices": total_choices,
                "risky_rate": risky_choices / total_choices,
            }
        )

    return trials


def fit_crra_parameters(
    trials: List[Dict[str, Any]], background_wealth: float
) -> Optional[Any]:
    """
    Fit CRRA parameters using maximum likelihood estimation.

    Uses multiple starting points and L-BFGS-B optimization to find
    the best-fitting CRRA parameters (r, beta) for the observed choice data.

    Args:
        trials: List of trial dictionaries with choice data
        background_wealth: Initial wealth assumption

    Returns:
        Optimization result object, or None if fitting fails

    Note:
        Uses multiple starting points to avoid local minima in optimization
    """
    # Use configuration constants for starting points and bounds
    starting_points = STARTING_POINTS

    best_result = None
    best_nll = np.inf

    for r0, beta0 in starting_points:
        try:
            result = minimize(
                negative_log_likelihood,
                x0=[r0, beta0],
                args=(trials, background_wealth),
                method="L-BFGS-B",
                bounds=[RISK_AVERSION_BOUNDS, CHOICE_SENSITIVITY_BOUNDS],
                options={
                    "maxiter": MAX_OPTIMIZATION_ITERATIONS,
                    "ftol": OPTIMIZATION_TOLERANCE,
                },
            )

            if result.success and result.fun < best_nll:
                best_result = result
                best_nll = result.fun

        except Exception as e:
            logger.warning(f"Optimization failed with start ({r0}, {beta0}): {e}")
            continue

    return best_result


def plot_individual_model(
    data: Dict[str, Any], output_dir: Path, background_wealth: float = BACKGROUND_WEALTH
) -> Optional[Path]:
    """
    Create individual plot for one model's CRRA analysis results.

    Generates a scatter plot showing observed choice probabilities vs.
    fitted logit model predictions, along with the fitted curve.

    Args:
        data: Analysis results dictionary containing parameters and trial data
        output_dir: Directory to save the plot
        background_wealth: Background wealth assumption for CRRA calculations

    Returns:
        Path to saved plot file, or None if plotting fails
    """
    trials = data["raw_trials"]
    r = data["parameters"]["risk_aversion_r"]
    beta = data["parameters"]["choice_sensitivity_beta"]
    model_name = data["model"]

    # Calculate EU differences and observed probabilities
    x_vals = []  # EU(risky) - EU(safe)
    y_vals = []  # Observed probability of choosing risky

    for trial in trials:
        eu_safe = expected_utility_safe(trial["safe_payoff"], background_wealth, r)
        eu_risky = expected_utility_risky(
            trial["risky_payoff"], trial["prob_win"], background_wealth, r
        )
        x_vals.append(eu_risky - eu_safe)
        y_vals.append(trial["risky_rate"])

    # Create smooth fitted curve
    if x_vals:
        x_grid = np.linspace(min(x_vals), max(x_vals), 200)
        y_fit = expit(beta * x_grid)  # Fitted logit predictions

        # Create plot
        plt.figure(figsize=FIGURE_SIZE_INDIVIDUAL)
        plt.scatter(
            x_vals,
            y_vals,
            alpha=0.7,
            s=80,
            color="steelblue",
            edgecolor="darkblue",
            linewidth=0.5,
            label="Observed Data",
        )
        plt.plot(x_grid, y_fit, "red", linewidth=3, label="Fitted Logit Model")

        # Styling
        plt.title(
            f"CRRA Analysis: O3 Mini (Opportunity Hunter)\n"
            f"Risk Aversion (r) = {r:.3f}, Choice Sensitivity (β) = {beta:.2f}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Expected Utility Difference: EU(Risky) - EU(Safe)", fontsize=12)
        plt.ylabel("Probability of Choosing Risky Option", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        # Save plot
        output_file = output_dir / "crra_individual_o3-mini_opportunity_hunter.png"
        plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Individual CRRA plot saved: {output_file}")
        return output_file

    return None


def plot_combined_models(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    background_wealth: float = BACKGROUND_WEALTH,
) -> Optional[Path]:
    """
    Create combined plot showing all models' CRRA analysis results.

    Generates a comparative plot showing fitted logit curves for all
    analyzed models on the same axes.

    Args:
        all_results: Dictionary mapping model names to analysis results
        output_dir: Directory to save the plot
        background_wealth: Background wealth assumption for CRRA calculations

    Returns:
        Path to saved plot file, or None if plotting fails
    """
    plt.figure(figsize=FIGURE_SIZE_COMBINED)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    for i, (model_name, data) in enumerate(all_results.items()):
        trials = data["raw_trials"]
        r = data["parameters"]["risk_aversion_r"]
        beta = data["parameters"]["choice_sensitivity_beta"]
        color = colors[i % len(colors)]

        # Calculate data points
        x_vals, y_vals = [], []
        for trial in trials:
            eu_safe = expected_utility_safe(trial["safe_payoff"], background_wealth, r)
            eu_risky = expected_utility_risky(
                trial["risky_payoff"], trial["prob_win"], background_wealth, r
            )
            x_vals.append(eu_risky - eu_safe)
            y_vals.append(trial["risky_rate"])

        if x_vals:
            # Fitted curve
            x_grid = np.linspace(min(x_vals), max(x_vals), 200)
            y_fit = expit(beta * x_grid)

            # Plot
            plt.plot(
                x_grid,
                y_fit,
                color=color,
                linewidth=3,
                label=f"{model_name} (r={r:.2f}, β={beta:.1f})",
            )
            plt.scatter(x_vals, y_vals, color=color, alpha=0.5, s=50, edgecolor="white")

    # Styling
    plt.title(
        "CRRA Risk Preferences Comparison Across AI Models",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Expected Utility Difference: EU(Risky) - EU(Safe)", fontsize=13)
    plt.ylabel("Probability of Choosing Risky Option", fontsize=13)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save combined plot
    combined_file = output_dir / "crra_combined_all_models.png"
    plt.savefig(combined_file, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"🎨 Combined CRRA plot saved: {combined_file}")
    return combined_file


def analyze_model(
    model_name: str, background_wealth: float = BACKGROUND_WEALTH
) -> Optional[Dict[str, Any]]:
    """
    Complete CRRA analysis for one model.

    Loads data, fits parameters, generates plots, and saves comprehensive results.

    Args:
        model_name: Name of the model to analyze
        background_wealth: Initial wealth assumption

    Returns:
        Analysis results dictionary, or None if analysis fails
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CRRA ANALYSIS: {model_name}")
    logger.info(f"{'='*60}")

    # File path (updated for new game format)
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()

    # Look specifically for o3-mini_opportunity_hunter_results.json file
    input_file = script_dir.parent / "results/o3-mini_opportunity_hunter_results.json"

    if not input_file.exists():
        logger.error(f"❌ Target file not found: {input_file}")
        return None

    logger.info(f"Loading data from: {input_file}")
    trials = load_and_process_data(input_file)

    if not trials:
        logger.error(f"❌ No valid data found for {model_name}")
        return None

    logger.info(f"✅ Loaded {len(trials)} valid scenarios")
    logger.info(f"📊 Background wealth: {background_wealth} tokens")

    # Fit parameters
    logger.info("\n🔍 Fitting CRRA parameters...")
    result = fit_crra_parameters(trials, background_wealth)

    if result is None or not result.success:
        logger.error(f"❌ Parameter fitting failed for {model_name}")
        return None

    r_hat, beta_hat = result.x
    nll = result.fun

    # Results
    logger.info(f"\n📈 FITTED PARAMETERS:")
    logger.info(f"   Risk Aversion (r): {r_hat:.4f}")
    logger.info(f"   Choice Sensitivity (β): {beta_hat:.2f}")
    logger.info(f"   Negative Log-Likelihood: {nll:.2f}")

    # Interpret risk aversion
    if r_hat < -0.5:
        risk_type = "Highly Risk Seeking"
    elif r_hat < 0:
        risk_type = "Risk Seeking"
    elif abs(r_hat) < 0.1:
        risk_type = "Risk Neutral"
    elif r_hat < 0.5:
        risk_type = "Mildly Risk Averse"
    elif r_hat < 1:
        risk_type = "Moderately Risk Averse"
    elif r_hat < 2:
        risk_type = "Risk Averse"
    else:
        risk_type = "Highly Risk Averse"

    logger.info(f"   Risk Type: {risk_type}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTDIR / f"crra_results_o3-mini_opportunity_hunter_{timestamp}.json"

    results = {
        "model": "o3-mini-opportunity-hunter",
        "timestamp": timestamp,
        "background_wealth": background_wealth,
        "parameters": {
            "risk_aversion_r": float(r_hat),
            "choice_sensitivity_beta": float(beta_hat),
        },
        "fit_quality": {
            "negative_log_likelihood": float(nll),
            "convergence_success": result.success,
        },
        "interpretation": {
            "risk_type": risk_type,
            "risk_aversion_level": float(r_hat),
        },
        "data_summary": {
            "trials_analyzed": len(trials),
            "total_choices": sum(t["total_choices"] for t in trials),
            "overall_risky_rate": sum(t["risky_choices"] for t in trials)
            / sum(t["total_choices"] for t in trials),
        },
        "raw_trials": trials,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"💾 Results saved: {output_file}")

    # Create individual plot
    plot_individual_model(results, OUTDIR, background_wealth)

    return results


def analyze_custom_file(
    file_path: Path, model_name: str, background_wealth: float = BACKGROUND_WEALTH
) -> Optional[Dict[str, Any]]:
    """
    Complete CRRA analysis for a custom data file.

    Allows analysis of data files from different sources or experimental conditions.

    Args:
        file_path: Path to the JSON data file
        model_name: Name/identifier for the model being analyzed
        background_wealth: Initial wealth assumption

    Returns:
        Analysis results dictionary, or None if analysis fails
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CRRA ANALYSIS: {model_name} ({file_path.name})")
    logger.info(f"{'='*60}")

    logger.info(f"Loading data from: {file_path}")
    trials = load_and_process_data(file_path)

    if not trials:
        logger.error(f"❌ No valid data found for {model_name}")
        return None

    logger.info(f"✅ Loaded {len(trials)} valid scenarios")
    logger.info(f"📊 Background wealth: {background_wealth} tokens")

    # Fit parameters
    logger.info("\n🔍 Fitting CRRA parameters...")
    result = fit_crra_parameters(trials, background_wealth)

    if result is None or not result.success:
        logger.error(f"❌ Parameter fitting failed for {model_name}")
        return None

    r_hat, beta_hat = result.x
    nll = result.fun

    # Results
    logger.info(f"\n📈 FITTED PARAMETERS:")
    logger.info(f"   Risk Aversion (r): {r_hat:.4f}")
    logger.info(f"   Choice Sensitivity (β): {beta_hat:.2f}")
    logger.info(f"   Negative Log Likelihood: {nll:.2f}")

    # Determine risk type
    if abs(r_hat) < 0.1:
        risk_type = "Risk Neutral"
    elif r_hat < 0:
        risk_type = "Risk Seeking"
    else:
        risk_type = "Risk Averse"

    logger.info(f"   Risk Type: {risk_type}")

    # Analyze choice patterns
    total_risky_choices = sum(t["risky_choices"] for t in trials)
    total_choices = sum(t["total_choices"] for t in trials)
    analysis = {
        "risky_rate": total_risky_choices / total_choices if total_choices > 0 else 0,
        "total_risky_choices": total_risky_choices,
        "total_choices": total_choices,
    }

    # Create comprehensive results
    results_data = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "parameters": {
            "risk_aversion_r": float(r_hat),
            "choice_sensitivity_beta": float(beta_hat),
        },
        "fit_quality": {
            "negative_log_likelihood": float(nll),
            "convergence_success": bool(result.success),
        },
        "interpretation": {"risk_type": risk_type, "risk_aversion_level": float(r_hat)},
        "data_summary": {
            "trials_analyzed": len(trials),
            "total_choices": total_choices,  # Use actual count, not assumption
            "overall_risky_rate": analysis["risky_rate"],
        },
        "raw_trials": trials,
    }

    # Save individual results
    output_file = (
        OUTDIR
        / f"crra_analysis_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"💾 Saved individual results to: {output_file}")

    # Create individual plots
    logger.info(f"🎨 Creating visualization for {model_name}...")
    plot_individual_model(results_data, OUTDIR, background_wealth)

    return results_data


def main() -> None:
    """
    Main CRRA analysis function with plotting.

    Orchestrates the complete analysis workflow including data loading,
    parameter fitting, visualization generation, and result summarization.
    """
    logger.info("🎯 ENHANCED CRRA ANALYSIS WITH VISUALIZATION")
    logger.info("=" * 70)
    logger.info(f"📊 Background wealth assumption: {BACKGROUND_WEALTH} tokens")
    logger.info(f"📁 Output directory: {OUTDIR}")
    logger.info("")

    if models_to_process:
        all_results = {}

        # Analyze each model
        for model in models_to_process:
            try:
                result = analyze_model(model)
                if result:
                    all_results[model] = result
                else:
                    logger.warning(f"⚠️  Failed to analyze {model}")
            except Exception as e:
                logger.error(f"❌ Error analyzing {model}: {e}")

        # Create combined plot if we have multiple successful analyses
        if len(all_results) > 1:
            logger.info(f"\n🎨 Creating combined visualization...")
            plot_combined_models(all_results, OUTDIR, BACKGROUND_WEALTH)

        # Final summary
        logger.info(f"\n🏁 CRRA ANALYSIS COMPLETE")
        logger.info(f"✅ Successfully analyzed: {len(all_results)} models")

        if all_results:
            logger.info(f"\n📊 CRRA RISK PREFERENCE SUMMARY:")
            logger.info("-" * 70)
            for model, data in all_results.items():
                r_val = data["parameters"]["risk_aversion_r"]
                beta_val = data["parameters"]["choice_sensitivity_beta"]
                risk_type = data["interpretation"]["risk_type"]
                risky_rate = data["data_summary"]["overall_risky_rate"]
                logger.info(
                    f"   {model:15s}: r={r_val:6.3f}, β={beta_val:5.1f} | "
                    f"{risk_type:20s} | {risky_rate:.1%} risky choices"
                )

            logger.info(f"\n💡 CRRA INTERPRETATION GUIDE:")
            logger.info(f"   • r < 0: Risk seeking behavior")
            logger.info(f"   • r ≈ 0: Risk neutral behavior")
            logger.info(f"   • r > 0: Risk averse behavior")
            logger.info(f"   • Higher β: More sensitive to utility differences")

    else:
        # Custom directory processing for persona experiment
        logger.info("🔄 Processing custom directory for persona experiment...")
        all_results = {}

        # Look for opportunity hunter results first
        persona_files = list(INPUT.glob("*_opportunity_hunter_results.json"))
        if persona_files:
            logger.info(f"📂 Found {len(persona_files)} persona experiment files")
            for file_path in persona_files:
                model_name = file_path.stem.replace("_opportunity_hunter_results", "")
                logger.info(f"🎭 Analyzing persona results for {model_name}...")
                result = analyze_custom_file(file_path, model_name)
                if result:
                    all_results[model_name] = result
        else:
            # Fallback to regular risk game results
            risk_files = list(INPUT.glob("*_risk_game_results.json"))
            logger.info(f"📂 Found {len(risk_files)} baseline experiment files")
            for file_path in risk_files:
                model_name = file_path.stem.replace("_risk_game_results", "")
                logger.info(f"🎯 Analyzing baseline results for {model_name}...")
                result = analyze_custom_file(file_path, model_name)
                if result:
                    all_results[model_name] = result

        # Create combined plot if we have multiple successful analyses
        if len(all_results) > 1:
            logger.info(f"\n🎨 Creating combined visualization...")
            plot_combined_models(all_results, OUTDIR, BACKGROUND_WEALTH)

        # Final summary
        logger.info(f"\n🏁 CUSTOM CRRA ANALYSIS COMPLETE")
        logger.info(f"✅ Successfully analyzed: {len(all_results)} files")

        if all_results:
            logger.info(f"\n📊 PERSONA EXPERIMENT CRRA RISK PREFERENCE SUMMARY:")
            logger.info("-" * 70)
            for model, data in all_results.items():
                r_val = data["parameters"]["risk_aversion_r"]
                beta_val = data["parameters"]["choice_sensitivity_beta"]
                risk_type = data["interpretation"]["risk_type"]
                risky_rate = data["data_summary"]["overall_risky_rate"]
                logger.info(
                    f"   {model:15s}: r={r_val:6.3f}, β={beta_val:5.1f} | "
                    f"{risk_type:20s} | {risky_rate:.1%} risky choices"
                )


if __name__ == "__main__":
    main()
