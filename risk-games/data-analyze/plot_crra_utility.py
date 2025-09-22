"""Plot CRRA utility functions from risk game result JSON files.

Scans the analysis directory for files named 'crra_results_*.json', extracts the
risk aversion parameter r ("risk_aversion_r") and plots the corresponding CRRA
utility function:

    U(w) = (w**(1-r))/(1-r)            if r != 1
    U(w) = ln(w)                       if r == 1

The script builds a wealth grid from (near) 0 up to a value slightly above the
largest payoff (safe or risky) appearing in the union of all JSON files plus
background wealth (if present). Saves a combined plot as 'crra_utility_functions.png'
inside the analysis directory by default.

Usage (PowerShell):
    python ambiguity-aversion/data-analyze/plot_crra_utility.py \
        --analysis-dir ambiguity-aversion/games_outputs/risk_game/analysis \
        --output crra_utility_functions.png

Optional arguments allow restricting to particular models.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_crra_files(analysis_dir: Path, include_models: List[str] | None) -> List[Dict[str, Any]]:
    files = sorted(analysis_dir.glob('crra_results_*.json'))
    results = []
    for f in files:
        try:
            with f.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
            model_name = data.get('model') or f.stem
            if include_models and model_name not in include_models:
                continue
            # Ensure required parameter exists
            params = data.get('parameters', {})
            if 'risk_aversion_r' not in params:
                continue
            results.append(data)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return results


def compute_max_wealth(datasets: List[Dict[str, Any]]) -> float:
    max_payoff = 0.0
    for d in datasets:
        background = d.get('background_wealth', 0) or 0
        raw_trials = d.get('raw_trials') or []
        for t in raw_trials:
            # Some truncated entries in provided JSONs only contain 'risky_rate'; skip.
            safe = t.get('safe_payoff') or 0
            risky = t.get('risky_payoff') or 0
            max_payoff = max(max_payoff, background + safe, background + risky)
    # Fallback if no trials parsed
    if max_payoff <= 0:
        max_payoff = 2000
    return max_payoff * 1.05  # add small margin


def crra_utility(w: np.ndarray, r: float) -> np.ndarray:
    # Handle r approximately 1 using log to avoid numerical issues
    if abs(r - 1.0) < 1e-8:
        return np.log(w)
    return (np.power(w, 1 - r) - 1) / (1 - r)


def main():
    parser = argparse.ArgumentParser(description="Plot CRRA utility functions from result JSON files.")
    parser.add_argument('--analysis-dir', type=Path, required=True,
                        help='Directory containing crra_results_*.json')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output PNG filename (default: crra_utility_functions.png in analysis dir)')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Optional subset of model names to include (e.g. gpt-4o gpt-5)')
    parser.add_argument('--min-wealth', type=float, default=1.0,
                        help='Minimum wealth value (default 1.0). Must be > 0 for log case.')
    parser.add_argument('--points', type=int, default=400,
                        help='Number of points in wealth grid (default 400)')
    args = parser.parse_args()

    analysis_dir: Path = args.analysis_dir
    if not analysis_dir.is_dir():
        raise SystemExit(f"Analysis directory not found: {analysis_dir}")

    datasets = load_crra_files(analysis_dir, args.models)
    if not datasets:
        raise SystemExit("No CRRA result JSON files found or matched.")

    max_w = compute_max_wealth(datasets)
    w_min = max(args.min_wealth, 1e-6)
    wealth = np.linspace(w_min, max_w, args.points)

    plt.figure(figsize=(8, 5))
    for d in datasets:
        r = d['parameters']['risk_aversion_r']
        model = d.get('model', 'unknown')
        U = crra_utility(wealth, r)
        # Normalize utility so that U(w_min)=0 for visual comparability
        U = U - U[0]
        plt.plot(wealth, U, label=f"{model} (r={r:.3f})")

    plt.xlabel('Wealth')
    plt.ylabel('Utility (CRRA, normalized)')
    plt.title('CRRA Utility Functions Across Models')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = args.output or (analysis_dir / 'crra_utility_functions.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    # Also optionally print a small table of risk aversion parameters
    print("\nModel risk aversion parameters:")
    for d in datasets:
        print(f"  {d.get('model','unknown')}: r={d['parameters']['risk_aversion_r']:.6f}")


if __name__ == '__main__':
    main()
