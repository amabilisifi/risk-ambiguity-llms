"""Plot CRRA utility functions (utility vs. wealth) from crra_analysis_*.json files.

This focuses on the 'analysis' JSON outputs (e.g., opportunity_hunter_analysis) that
contain the fitted parameter risk_aversion_r (and possibly beta). We produce a
plot of the CRRA utility function for each model:

    U(w) = (w**(1-r) - 1)/(1-r)   if r != 1
    U(w) = ln(w)                  if r == 1

We normalize each curve so U(w_min) = 0 for comparability and optionally also
scale by its max absolute value if --scale is provided.

Example (PowerShell):
    python ambiguity-aversion/data-analyze/plot_crra_utility_from_analysis.py \
        --analysis-dir ambiguity-aversion/games_outputs/risk_game/opportunity_hunter_analysis

Optional: specify pattern (default crra_analysis_*.json) or include subset of models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def crra_utility_array(w: np.ndarray, r: float) -> np.ndarray:
    if abs(r - 1.0) < 1e-8:
        return np.log(w)
    return (np.power(w, 1 - r) - 1) / (1 - r)


def load_files(analysis_dir: Path, pattern: str, include_models: List[str] | None) -> List[Dict[str, Any]]:
    data = []
    for file in sorted(analysis_dir.glob(pattern)):
        try:
            with file.open('r', encoding='utf-8') as fh:
                content = json.load(fh)
            model = content.get('model') or file.stem
            if include_models and model not in include_models:
                continue
            params = content.get('parameters', {})
            if 'risk_aversion_r' not in params:
                continue
            content['__filepath'] = file
            data.append(content)
        except Exception as e:
            print(f"Skipping {file.name}: {e}")
    return data


def wealth_upper_bound(datasets: List[Dict[str, Any]], background_wealth: float) -> float:
    max_payoff = 0.0
    for d in datasets:
        trials = d.get('raw_trials') or []
        for t in trials:
            safe = t.get('safe_payoff') or 0
            risky = t.get('risky_payoff') or 0
            max_payoff = max(max_payoff, background_wealth + safe, background_wealth + risky)
    if max_payoff <= 0:
        max_payoff = background_wealth + 2000
    return max_payoff * 1.05


def main():
    parser = argparse.ArgumentParser(description='Plot CRRA utility vs wealth from analysis JSON files.')
    parser.add_argument('--analysis-dir', type=Path, required=True, help='Directory containing crra_analysis_*.json')
    parser.add_argument('--pattern', type=str, default='crra_analysis_*.json', help='Glob pattern for files (default: crra_analysis_*.json)')
    parser.add_argument('--models', nargs='*', default=None, help='Optional subset of model names to include')
    parser.add_argument('--background-wealth', type=float, default=100.0, help='Background wealth assumption (default 100)')
    parser.add_argument('--min-wealth', type=float, default=1.0, help='Minimum wealth for grid (must be >0)')
    parser.add_argument('--points', type=int, default=400, help='Number of grid points (default 400)')
    parser.add_argument('--scale', action='store_true', help='After normalization, scale each curve by its max absolute value to fit within [0,1].')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG filename (default derived)')
    args = parser.parse_args()

    if not args.analysis_dir.is_dir():
        raise SystemExit(f"Directory not found: {args.analysis_dir}")

    datasets = load_files(args.analysis_dir, args.pattern, args.models)
    if not datasets:
        raise SystemExit('No matching analysis files with risk_aversion_r found.')

    w_max = wealth_upper_bound(datasets, args.background_wealth)
    w_min = max(args.min_wealth, 1e-6)
    wealth = np.linspace(w_min, w_max, args.points)

    plt.figure(figsize=(8, 5.5))
    for d in datasets:
        r = d['parameters']['risk_aversion_r']
        model = d.get('model', 'unknown')
        U = crra_utility_array(wealth, r)
        # Normalize start point
        U = U - U[0]
        if args.scale:
            max_abs = np.max(np.abs(U))
            if max_abs > 0:
                U = U / max_abs
        plt.plot(wealth, U, linewidth=2, label=f"{model} (r={r:.3f})")

    plt.xlabel('Wealth')
    ylabel = 'Utility (CRRA, normalized' + (' & scaled' if args.scale else '') + ')'
    plt.ylabel(ylabel)
    plt.title('CRRA Utility Functions (Analysis Files)')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_file = args.output or (args.analysis_dir / 'crra_utility_functions_analysis.png')
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")

    print('\nModel parameters:')
    for d in datasets:
        print(f"  {d.get('model','unknown')}: r={d['parameters']['risk_aversion_r']:.6f}")


if __name__ == '__main__':
    main()
