"""
CARA (Constant Absolute Risk Aversion) Analysis for AI Risk Preferences

This script analyzes AI language models' risk preferences using Constant Absolute Risk Aversion
(CARA) utility functions. It fits CARA parameters to choice data from risk-reward experiments
and generates comprehensive visualizations and statistical analyses.

WHAT THIS SCRIPT DOES:
- Loads and processes risk game choice data from JSON files
- Fits CARA utility function parameters using maximum likelihood estimation
- Generates individual and comparative plots of risk preferences
- Provides statistical summaries and risk preference interpretations

CARA MODEL:
The CARA utility function is: U(x) = -exp(-α*x)/α for α ≠ 0, U(x) = x for α = 0
- α < 0: Risk seeking behavior
- α ≈ 0: Risk neutral behavior
- α > 0: Risk averse behavior (higher α = more risk averse)

REQUIREMENTS:
- Python 3.8+
- Required packages: numpy, scipy, matplotlib, collections, json, pathlib
- Risk game experiment data in JSON format

USAGE:
1. Run from risk-games/data-analyze/ directory
2. Default: python fit_cara.py (analyzes available models)
3. Custom file: python fit_cara.py /path/to/results.json
4. Custom output: python fit_cara.py input.json output_dir

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

# File paths and directories
DEFAULT_INPUT_DIR = "../results"  # Relative to script location
DEFAULT_OUTPUT_DIR = "analysis"  # Relative to script location
# TODO: Set your input/output directories here if different from defaults

# Analysis parameters
RISK_AVERSION_BOUNDS = (-1.0, 5.0)  # Allowable range for α parameter
CHOICE_SENSITIVITY_BOUNDS = (0.01, 100.0)  # Allowable range for β parameter (> 0)

# Optimization settings
MAX_OPTIMIZATION_ITERATIONS = 2000
OPTIMIZATION_TOLERANCE = 1e-9
STARTING_POINTS = [
    (0.01, 1.0),  # Very low risk aversion
    (0.1, 2.0),  # Low risk aversion
    (0.5, 1.0),  # Moderate risk aversion
    (1.0, 5.0),  # High risk aversion
    (0.001, 0.5),  # Near risk neutral
    (-0.1, 1.0),  # Risk seeking
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

logger.info("🎯 CARA Analysis for AI Risk Preferences")
logger.info("=" * 60)
logger.info(f"📁 Output directory: {OUTDIR}")
logger.info(f"📅 Analysis timestamp: {STAMP}")


def cara_utility(payoff: float, alpha: float) -> float:
    """
    Compute Constant Absolute Risk Aversion (CARA) utility for a payoff.

    The CARA utility function is: U(x) = -exp(-α*x) / α for α ≠ 0, U(x) = x for α = 0

    Args:
        payoff: The monetary payoff amount
        alpha: Risk aversion parameter (α)
            - α < 0: Risk seeking
            - α ≈ 0: Risk neutral
            - α > 0: Risk averse (higher values = more risk averse)

    Returns:
        Utility value for the given payoff

    Note:
        Includes numerical stability measures to prevent overflow/underflow
        in exponential calculations.
    """
    if abs(alpha) < 1e-10:  # α ≈ 0 (risk neutral)
        return payoff

    # Numerical stability for exponential
    exp_arg = -alpha * payoff
    if exp_arg > 700:  # Prevent overflow
        return 0.0
    elif exp_arg < -700:  # Prevent underflow
        return -1e10  # Large negative utility

    # Standard CARA formula: U(x) = -exp(-αx)/α
    return -math.exp(exp_arg) / alpha


def expected_utility_safe(safe_payoff: float, alpha: float) -> float:
    """Compute expected utility of the safe option."""
    return cara_utility(safe_payoff, alpha)


def expected_utility_risky(risky_payoff: float, prob_win: float, alpha: float) -> float:
    """
    Compute expected utility of the risky option.

    The risky option pays risky_payoff with probability prob_win,
    and 0 with probability (1 - prob_win).

    Args:
        risky_payoff: Payoff amount if the risky option succeeds
        prob_win: Probability of success for the risky option
        alpha: Risk aversion parameter

    Returns:
        Expected utility of the risky option
    """
    # Win scenario: get risky_payoff
    # Lose scenario: get 0
    eu_win = cara_utility(risky_payoff, alpha)
    eu_lose = cara_utility(0.0, alpha)

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


def negative_log_likelihood(params, trials):
    """Negative log-likelihood for parameter estimation"""
    alpha, beta = params

    # Parameter bounds check
    if beta <= 0:
        return 1e10

    nll = 0.0

    for trial in trials:
        try:
            # Calculate expected utilities using CARA
            eu_safe = expected_utility_safe(trial["safe_payoff"], alpha)
            eu_risky = expected_utility_risky(
                trial["risky_payoff"], trial["prob_win"], alpha
            )

            # Probability of choosing risky
            p_risky = choice_probability(eu_risky, eu_safe, beta)
            p_risky = np.clip(p_risky, 1e-12, 1 - 1e-12)  # Prevent log(0)

            # Add to log-likelihood
            m, n = trial["risky_choices"], trial["total_choices"]
            nll += -(m * np.log(p_risky) + (n - m) * np.log(1 - p_risky))

        except (OverflowError, ValueError, ZeroDivisionError) as e:
            print(f"Numerical error in trial {trial.get('id', 'unknown')}: {e}")
            return 1e10

    return nll


def load_and_process_data(input_file):
    """Load and process data from improved game format"""
    try:
        with open(input_file, encoding="utf-8") as f:
            raw_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {input_file}: {e}")
        return None

    # Count choices by scenario
    choice_counts = collections.Counter()
    scenario_info = {}

    for decision in raw_data:
        # Handle both old and new data formats
        scenario_id = decision.get("scenario_id", decision.get("id"))
        choice = decision.get("choice")

        # Skip invalid entries
        if not choice or choice not in ["safe", "risky"]:
            continue

        # Store scenario parameters (flexible for both formats)
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
            print(f"Skipping invalid scenario {scenario_id}: {info}")
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


def fit_cara_parameters(trials):
    """Fit CARA parameters using maximum likelihood estimation"""

    # Multiple starting points to avoid local minima
    starting_points = [
        (0.01, 1.0),  # Very low risk aversion
        (0.1, 2.0),  # Low risk aversion
        (0.5, 1.0),  # Moderate risk aversion
        (1.0, 5.0),  # High risk aversion
        (0.001, 0.5),  # Near risk neutral
        (-0.1, 1.0),  # Risk seeking
    ]

    best_result = None
    best_nll = np.inf

    for alpha0, beta0 in starting_points:
        try:
            result = minimize(
                negative_log_likelihood,
                x0=[alpha0, beta0],
                args=(trials,),
                method="L-BFGS-B",
                bounds=[(-1.0, 5.0), (0.01, 100.0)],  # Allow risk seeking, beta > 0
                options={"maxiter": 2000, "ftol": 1e-9},
            )

            if result.success and result.fun < best_nll:
                best_result = result
                best_nll = result.fun

        except Exception as e:
            print(f"Optimization failed with start ({alpha0}, {beta0}): {e}")
            continue

    return best_result


def plot_individual_model(data, output_dir):
    """Create individual plot for one model"""
    trials = data["raw_trials"]
    alpha = data["parameters"]["risk_aversion_alpha"]
    beta = data["parameters"]["choice_sensitivity_beta"]
    model_name = data["model"]

    # Calculate EU differences and observed probabilities
    x_vals = []  # EU(risky) - EU(safe)
    y_vals = []  # Observed probability of choosing risky

    for trial in trials:
        eu_safe = expected_utility_safe(trial["safe_payoff"], alpha)
        eu_risky = expected_utility_risky(
            trial["risky_payoff"], trial["prob_win"], alpha
        )
        x_vals.append(eu_risky - eu_safe)
        y_vals.append(trial["risky_rate"])

    # Create smooth fitted curve
    if x_vals:
        x_grid = np.linspace(min(x_vals), max(x_vals), 200)
        y_fit = expit(beta * x_grid)  # Fitted logit predictions

        # Create plot
        plt.figure(figsize=(10, 7))
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
            f"CARA Analysis: O3 Mini (Opportunity Hunter)\n"
            f"Risk Aversion (α) = {alpha:.4f}, Choice Sensitivity (β) = {beta:.2f}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Expected Utility Difference: EU(Risky) - EU(Safe)", fontsize=12)
        plt.ylabel("Probability of Choosing Risky Option", fontsize=12)
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        # Save plot
        output_file = output_dir / "cara_individual_o3-mini_opportunity_hunter.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Individual CARA plot saved: {output_file}")
        return output_file

    return None


def plot_combined_models(all_results, output_dir):
    """Create combined plot showing all models"""
    plt.figure(figsize=(14, 9))

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
        alpha = data["parameters"]["risk_aversion_alpha"]
        beta = data["parameters"]["choice_sensitivity_beta"]
        color = colors[i % len(colors)]

        # Calculate data points
        x_vals, y_vals = [], []
        for trial in trials:
            eu_safe = expected_utility_safe(trial["safe_payoff"], alpha)
            eu_risky = expected_utility_risky(
                trial["risky_payoff"], trial["prob_win"], alpha
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
                label=f"{model_name} (α={alpha:.3f}, β={beta:.1f})",
            )
            plt.scatter(x_vals, y_vals, color=color, alpha=0.5, s=50, edgecolor="white")

    # Styling
    plt.title(
        "CARA Risk Preferences Comparison Across AI Models",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Expected Utility Difference: EU(Risky) - EU(Safe)", fontsize=13)
    plt.ylabel("Probability of Choosing Risky Option", fontsize=13)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save combined plot
    combined_file = output_dir / "cara_combined_all_models.png"
    plt.savefig(combined_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🎨 Combined CARA plot saved: {combined_file}")
    return combined_file


def analyze_model(model_name):
    """Complete CARA analysis for one model"""

    print(f"\n{'='*60}")
    print(f"CARA ANALYSIS: O3 Mini (Opportunity Hunter)")
    print(f"{'='*60}")

    # File path (handles both old and new formats)
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()

    # Look specifically for o3-mini_opportunity_hunter_results.json file
    input_file = script_dir.parent / "results/o3-mini_opportunity_hunter_results.json"

    if not input_file.exists():
        print(f"❌ Target file not found: {input_file}")
        return None

    print(f"Loading data from: {input_file}")
    trials = load_and_process_data(input_file)

    if not trials:
        print(f"❌ No valid data found for {model_name}")
        return None

    print(f"✅ Loaded {len(trials)} valid scenarios")

    # Fit parameters
    print("\n🔍 Fitting CARA parameters...")
    result = fit_cara_parameters(trials)

    if result is None or not result.success:
        print(f"❌ Parameter fitting failed for {model_name}")
        return None

    alpha_hat, beta_hat = result.x
    nll = result.fun

    # Results
    print(f"\n📈 FITTED PARAMETERS:")
    print(f"   Risk Aversion (α): {alpha_hat:.6f}")
    print(f"   Choice Sensitivity (β): {beta_hat:.2f}")
    print(f"   Negative Log-Likelihood: {nll:.2f}")

    # Interpret risk aversion
    if alpha_hat < -0.1:
        risk_type = "Risk Seeking"
    elif abs(alpha_hat) < 0.01:
        risk_type = "Risk Neutral"
    elif alpha_hat < 0.1:
        risk_type = "Mildly Risk Averse"
    elif alpha_hat < 0.5:
        risk_type = "Moderately Risk Averse"
    elif alpha_hat < 1.0:
        risk_type = "Risk Averse"
    else:
        risk_type = "Highly Risk Averse"

    print(f"   Risk Type: {risk_type}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTDIR / f"cara_results_o3-mini_opportunity_hunter_{timestamp}.json"

    results = {
        "model": "o3-mini-opportunity-hunter",
        "timestamp": timestamp,
        "parameters": {
            "risk_aversion_alpha": float(alpha_hat),
            "choice_sensitivity_beta": float(beta_hat),
        },
        "fit_quality": {
            "negative_log_likelihood": float(nll),
            "convergence_success": result.success,
        },
        "interpretation": {
            "risk_type": risk_type,
            "risk_aversion_level": float(alpha_hat),
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

    print(f"💾 Results saved: {output_file}")

    # Create individual plot
    plot_individual_model(results, OUTDIR)

    return results


def analyze_custom_file(file_path, model_name):
    """Complete CARA analysis for a custom file"""

    print(f"\n{'='*60}")
    print(f"CARA ANALYSIS: {model_name} ({file_path.name})")
    print(f"{'='*60}")

    print(f"Loading data from: {file_path}")
    trials = load_and_process_data(file_path)

    if not trials:
        print(f"❌ No valid data found for {model_name}")
        return None

    print(f"✅ Loaded {len(trials)} valid scenarios")

    # Fit parameters
    print("\n🔍 Fitting CARA parameters...")
    result = fit_cara_parameters(trials)

    if result is None or not result.success:
        print(f"❌ Parameter fitting failed for {model_name}")
        return None

    alpha_hat, beta_hat = result.x
    nll = result.fun

    # Results
    print(f"\n📈 FITTED PARAMETERS:")
    print(f"   Risk Aversion (α): {alpha_hat:.4f}")
    print(f"   Choice Sensitivity (β): {beta_hat:.2f}")
    print(f"   Negative Log Likelihood: {nll:.2f}")

    # Determine risk type
    if abs(alpha_hat) < 0.01:
        risk_type = "Risk Neutral"
    elif alpha_hat < 0:
        risk_type = "Risk Seeking"
    else:
        risk_type = "Risk Averse"

    print(f"   Risk Type: {risk_type}")

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
        "timestamp": STAMP,
        "parameters": {
            "risk_aversion_alpha": float(alpha_hat),
            "choice_sensitivity_beta": float(beta_hat),
        },
        "fit_quality": {
            "negative_log_likelihood": float(nll),
            "convergence_success": bool(result.success),
        },
        "interpretation": {
            "risk_type": risk_type,
            "risk_aversion_level": float(alpha_hat),
        },
        "data_summary": {
            "trials_analyzed": len(trials),
            "total_choices": total_choices,
            "overall_risky_rate": analysis["risky_rate"],
        },
        "raw_trials": trials,
    }

    # Save individual results
    output_file = OUTDIR / f"cara_analysis_{model_name}_{STAMP}.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"💾 Saved individual results to: {output_file}")

    # Create individual plots
    print(f"🎨 Creating visualization for {model_name}...")
    plot_individual_model(results_data, OUTDIR)

    return results_data


def main():
    """Main analysis function with plotting"""

    print("🎯 PROPER CARA ANALYSIS WITH VISUALIZATION")
    print("=" * 70)
    print(f"📁 Output directory: {OUTDIR}")
    print()

    if models_to_process:
        all_results = {}

        # Analyze each model
        for model in models_to_process:
            try:
                result = analyze_model(model)
                if result:
                    all_results[model] = result
                else:
                    print(f"⚠️  Failed to analyze {model}")
            except Exception as e:
                print(f"❌ Error analyzing {model}: {e}")

        # Create combined plot if we have multiple successful analyses
        if len(all_results) > 1:
            print(f"\n🎨 Creating combined visualization...")
            plot_combined_models(all_results, OUTDIR)

        # Final summary
        print(f"\n🏁 CARA ANALYSIS COMPLETE")
        print(f"✅ Successfully analyzed: {len(all_results)} models")

        if all_results:
            print(f"\n📊 CARA RISK PREFERENCE SUMMARY:")
            print("-" * 70)
            for model, data in all_results.items():
                alpha_val = data["parameters"]["risk_aversion_alpha"]
                beta_val = data["parameters"]["choice_sensitivity_beta"]
                risk_type = data["interpretation"]["risk_type"]
                risky_rate = data["data_summary"]["overall_risky_rate"]
                print(
                    f"   {model:15s}: α={alpha_val:8.4f}, β={beta_val:5.1f} | "
                    f"{risk_type:20s} | {risky_rate:.1%} risky choices"
                )

            print(f"\n💡 CARA INTERPRETATION GUIDE:")
            print(f"   • α < 0: Risk seeking behavior")
            print(f"   • α ≈ 0: Risk neutral behavior")
            print(f"   • α > 0: Risk averse behavior")
            print(f"   • Higher α: More risk averse")
            print(f"   • Higher β: More sensitive to utility differences")

    else:
        # Custom directory processing for persona experiment
        print("🔄 Processing custom directory for persona experiment...")
        all_results = {}

        # Look for opportunity hunter results first
        persona_files = list(INPUT.glob("*_opportunity_hunter_results.json"))
        if persona_files:
            print(f"📂 Found {len(persona_files)} persona experiment files")
            for file_path in persona_files:
                model_name = file_path.stem.replace("_opportunity_hunter_results", "")
                print(f"🎭 Analyzing persona results for {model_name}...")
                result = analyze_custom_file(file_path, model_name)
                if result:
                    all_results[model_name] = result
        else:
            # Fallback to regular risk game results
            risk_files = list(INPUT.glob("*_risk_game_results.json"))
            print(f"📂 Found {len(risk_files)} baseline experiment files")
            for file_path in risk_files:
                model_name = file_path.stem.replace("_risk_game_results", "")
                print(f"🎯 Analyzing baseline results for {model_name}...")
                result = analyze_custom_file(file_path, model_name)
                if result:
                    all_results[model_name] = result

        # Create combined plot if we have multiple successful analyses
        if len(all_results) > 1:
            print(f"\n🎨 Creating combined visualization...")
            plot_combined_models(all_results, OUTDIR)

        # Final summary
        print(f"\n🏁 CUSTOM CARA ANALYSIS COMPLETE")
        print(f"✅ Successfully analyzed: {len(all_results)} files")

        if all_results:
            print(f"\n📊 PERSONA EXPERIMENT CARA RISK PREFERENCE SUMMARY:")
            print("-" * 70)
            for model, data in all_results.items():
                alpha_val = data["parameters"]["risk_aversion_alpha"]
                beta_val = data["parameters"]["choice_sensitivity_beta"]
                risk_type = data["interpretation"]["risk_type"]
                risky_rate = data["data_summary"]["overall_risky_rate"]
                print(
                    f"   {model:15s}: α={alpha_val:8.4f}, β={beta_val:5.1f} | "
                    f"{risk_type:20s} | {risky_rate:.1%} risky choices"
                )


if __name__ == "__main__":
    main()
