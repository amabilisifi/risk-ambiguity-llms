"""
Fixed CRRA Analysis for Improved Risk Game
Works with new data structure and includes comprehensive plotting
"""

import collections
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Configuration - Only analyze O3 Mini Opportunity Hunter results
AVAILABLE_MODELS = ["o3-mini-opportunity-hunter"]
BACKGROUND_WEALTH = 100  # Must match the game assumption

# Parse command line arguments
if len(sys.argv) > 1 and sys.argv[1] in AVAILABLE_MODELS:
    models_to_process = [sys.argv[1]]
elif len(sys.argv) > 1:
    INPUT = Path(sys.argv[1])
    models_to_process = None
else:
    models_to_process = AVAILABLE_MODELS.copy()

OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)


def crra_utility(wealth, r):
    """
    CRRA utility function: U(w) = (w^(1-r) - 1)/(1-r) for r ≠ 1, ln(w) for r = 1
    """
    if wealth <= 0:
        return -np.inf

    if abs(r - 1) < 1e-8:  # r ≈ 1 (log utility)
        return np.log(wealth)
    else:
        return (wealth ** (1 - r) - 1) / (1 - r)


def expected_utility_safe(safe_payoff, background_wealth, r):
    """Expected utility of safe option"""
    return crra_utility(background_wealth + safe_payoff, r)


def expected_utility_risky(risky_payoff, prob_win, background_wealth, r):
    """Expected utility of risky option"""
    wealth_if_win = background_wealth + risky_payoff
    wealth_if_lose = background_wealth  # Win 0, keep background wealth

    eu_win = crra_utility(wealth_if_win, r)
    eu_lose = crra_utility(wealth_if_lose, r)

    return prob_win * eu_win + (1 - prob_win) * eu_lose


def choice_probability(eu_risky, eu_safe, beta):
    """Probability of choosing risky option (logit model)"""
    utility_diff = eu_risky - eu_safe
    utility_diff = np.clip(utility_diff, -500, 500)  # Prevent overflow
    return expit(beta * utility_diff)


def negative_log_likelihood(params, trials, background_wealth):
    """Negative log-likelihood for parameter estimation"""
    r, beta = params

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


def fit_crra_parameters(trials, background_wealth):
    """Fit CRRA parameters using maximum likelihood estimation"""

    # Multiple starting points to avoid local minima
    starting_points = [
        (0.5, 1.0),  # Moderate risk aversion
        (1.0, 2.0),  # Log utility
        (2.0, 0.5),  # High risk aversion
        (0.1, 5.0),  # Risk seeking
        (1.5, 10.0),  # High risk aversion, high beta
        (-0.5, 1.0),  # Risk seeking
    ]

    best_result = None
    best_nll = np.inf

    for r0, beta0 in starting_points:
        try:
            result = minimize(
                negative_log_likelihood,
                x0=[r0, beta0],
                args=(trials, background_wealth),
                method="L-BFGS-B",
                bounds=[(-3.0, 5.0), (0.01, 100.0)],  # Wider r bounds, beta > 0
                options={"maxiter": 3000, "ftol": 1e-9},
            )

            if result.success and result.fun < best_nll:
                best_result = result
                best_nll = result.fun

        except Exception as e:
            print(f"Optimization failed with start ({r0}, {beta0}): {e}")
            continue

    return best_result


def plot_individual_model(data, output_dir, background_wealth=BACKGROUND_WEALTH):
    """Create individual plot for one model"""
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
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Individual plot saved: {output_file}")
        return output_file

    return None


def plot_combined_models(all_results, output_dir, background_wealth=BACKGROUND_WEALTH):
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
    plt.savefig(combined_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🎨 Combined plot saved: {combined_file}")
    return combined_file


def analyze_model(model_name, background_wealth=BACKGROUND_WEALTH):
    """Complete CRRA analysis for one model"""

    print(f"\n{'='*60}")
    print(f"ANALYZING: O3 Mini (Opportunity Hunter)")
    print(f"{'='*60}")

    # File path (updated for new game format)
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
    print(f"📊 Background wealth: {background_wealth} tokens")

    # Fit parameters
    print("\n🔍 Fitting CRRA parameters...")
    result = fit_crra_parameters(trials, background_wealth)

    if result is None or not result.success:
        print(f"❌ Parameter fitting failed for {model_name}")
        return None

    r_hat, beta_hat = result.x
    nll = result.fun

    # Results
    print(f"\n📈 FITTED PARAMETERS:")
    print(f"   Risk Aversion (r): {r_hat:.4f}")
    print(f"   Choice Sensitivity (β): {beta_hat:.2f}")
    print(f"   Negative Log-Likelihood: {nll:.2f}")

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

    print(f"   Risk Type: {risk_type}")

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

    print(f"💾 Results saved: {output_file}")

    # Create individual plot
    plot_individual_model(results, OUTDIR, background_wealth)

    return results


def analyze_custom_file(file_path, model_name, background_wealth=BACKGROUND_WEALTH):
    """Complete CRRA analysis for a custom file"""

    print(f"\n{'='*60}")
    print(f"ANALYZING: {model_name} ({file_path.name})")
    print(f"{'='*60}")

    print(f"Loading data from: {file_path}")
    trials = load_and_process_data(file_path)

    if not trials:
        print(f"❌ No valid data found for {model_name}")
        return None

    print(f"✅ Loaded {len(trials)} valid scenarios")
    print(f"📊 Background wealth: {background_wealth} tokens")

    # Fit parameters
    print("\n🔍 Fitting CRRA parameters...")
    result = fit_crra_parameters(trials, background_wealth)

    if result is None or not result.success:
        print(f"❌ Parameter fitting failed for {model_name}")
        return None

    r_hat, beta_hat = result.x
    nll = result.fun

    # Results
    print(f"\n📈 FITTED PARAMETERS:")
    print(f"   Risk Aversion (r): {r_hat:.4f}")
    print(f"   Choice Sensitivity (β): {beta_hat:.2f}")
    print(f"   Negative Log Likelihood: {nll:.2f}")

    # Determine risk type
    if abs(r_hat) < 0.1:
        risk_type = "Risk Neutral"
    elif r_hat < 0:
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
            "total_choices": len(trials) * 5,  # Assuming 5 trials per scenario
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

    print(f"💾 Saved individual results to: {output_file}")

    # Create individual plots
    print(f"🎨 Creating visualization for {model_name}...")
    plot_individual_model(results_data, OUTDIR, background_wealth)

    return results_data


def main():
    """Main analysis function with plotting"""

    print("🎯 ENHANCED CRRA ANALYSIS WITH VISUALIZATION")
    print("=" * 70)
    print(f"📊 Background wealth assumption: {BACKGROUND_WEALTH} tokens")
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
            plot_combined_models(all_results, OUTDIR, BACKGROUND_WEALTH)

        # Final summary
        print(f"\n🏁 ANALYSIS COMPLETE")
        print(f"✅ Successfully analyzed: {len(all_results)} models")

        if all_results:
            print(f"\n📊 RISK PREFERENCE SUMMARY:")
            print("-" * 70)
            for model, data in all_results.items():
                r_val = data["parameters"]["risk_aversion_r"]
                beta_val = data["parameters"]["choice_sensitivity_beta"]
                risk_type = data["interpretation"]["risk_type"]
                risky_rate = data["data_summary"]["overall_risky_rate"]
                print(
                    f"   {model:15s}: r={r_val:6.3f}, β={beta_val:5.1f} | "
                    f"{risk_type:20s} | {risky_rate:.1%} risky choices"
                )

            print(f"\n💡 INTERPRETATION GUIDE:")
            print(f"   • r < 0: Risk seeking behavior")
            print(f"   • r ≈ 0: Risk neutral behavior")
            print(f"   • r > 0: Risk averse behavior")
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
            plot_combined_models(all_results, OUTDIR, BACKGROUND_WEALTH)

        # Final summary
        print(f"\n🏁 CUSTOM ANALYSIS COMPLETE")
        print(f"✅ Successfully analyzed: {len(all_results)} files")

        if all_results:
            print(f"\n📊 PERSONA EXPERIMENT RISK PREFERENCE SUMMARY:")
            print("-" * 70)
            for model, data in all_results.items():
                r_val = data["parameters"]["risk_aversion_r"]
                beta_val = data["parameters"]["choice_sensitivity_beta"]
                risk_type = data["interpretation"]["risk_type"]
                risky_rate = data["data_summary"]["overall_risky_rate"]
                print(
                    f"   {model:15s}: r={r_val:6.3f}, β={beta_val:5.1f} | "
                    f"{risk_type:20s} | {risky_rate:.1%} risky choices"
                )


if __name__ == "__main__":
    main()
