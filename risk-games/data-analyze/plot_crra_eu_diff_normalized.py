"""Plot normalized Expected Utility differences vs. probability of choosing risky.

Loads CRRA result or analysis JSON files (pattern configurable) containing:
  - parameters.risk_aversion_r
  - parameters.choice_sensitivity_beta (optional; if absent, fitted curve omitted)
  - raw_trials with fields: prob_win, safe_payoff, risky_payoff, risky_rate

For each model/file we compute for every trial:
   x = EU_risky - EU_safe
   y = risky_rate
Then normalize x per model so curves are comparable on the same plot.

Normalization methods (choose with --norm):
    maxabs      : x' = x / max(|x|) (default)
    zscore      : x' = (x - mean(x)) / std(x)
    globalmax   : x' = x / global_max_abs_over_all_models
    robust      : x' = (x - median(x)) / (1.4826*MAD)
    logmaxabs   : x' = sign(x)*log(1+|x|)/log(1+max|x|)
    minmax      : x' = 2*(x - min)/(max - min) - 1  (maps each model range to [-1,1])
    rank        : x' = 2*(rank(x)/(n-1)) - 1        (monotonic, outlier-robust)
    pooledz     : Global (across models) z-score
    beta        : x' = beta * x  (rescales so logistic slopes comparable; if beta missing leaves unchanged)

Options:
    --clip-pct P     Winsorize EU diffs at P and 100-P percentiles before normalizing
    --export-csv F   Export per-point data to CSV
    --curve-samples N Adjust resolution of logistic curve (default 300)
    --raw-x-hist     Also save histogram of raw EU differences

The y-axis (probability of choosing risky) is already in [0,1], so unchanged.
We overlay the fitted logistic curve using the model's beta parameter, adjusted
for the normalization scaling. If original p = sigmoid(beta * x), and x = s * x',
then p = sigmoid(beta * s * x').

Example usage (PowerShell):
  python ambiguity-aversion/data-analyze/plot_crra_eu_diff_normalized.py \
      --analysis-dir ambiguity-aversion/games_outputs/risk_game/analysis \
      --pattern crra_results_*.json --norm maxabs

  python ambiguity-aversion/data-analyze/plot_crra_eu_diff_normalized.py \
      --analysis-dir ambiguity-aversion/games_outputs/risk_game/opportunity_hunter_analysis \
      --pattern crra_analysis_*.json --norm globalmax
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import csv


def crra_utility(w: float | np.ndarray, r: float) -> float | np.ndarray:
    w_arr = np.asarray(w)
    # Avoid log issues
    w_arr = np.where(w_arr <= 0, np.nan, w_arr)
    if abs(r - 1.0) < 1e-8:
        return np.log(w_arr)
    return (np.power(w_arr, 1 - r) - 1) / (1 - r)


def expected_utility_diff(trial: Dict[str, Any], r: float, background: float) -> float:
    try:
        p = trial['prob_win']
        safe = trial['safe_payoff']
        risky = trial['risky_payoff']
    except KeyError:
        return np.nan
    eu_safe = crra_utility(background + safe, r)
    eu_risky_win = crra_utility(background + risky, r)
    eu_risky_lose = crra_utility(background, r)
    eu_risky = p * eu_risky_win + (1 - p) * eu_risky_lose
    return float(eu_risky - eu_safe)


def load_files(directory: Path, pattern: str) -> List[Dict[str, Any]]:
    data = []
    for f in sorted(directory.glob(pattern)):
        try:
            with f.open('r', encoding='utf-8') as fh:
                content = json.load(fh)
            params = content.get('parameters', {})
            if 'risk_aversion_r' not in params:
                continue
            content['__file'] = f
            data.append(content)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return data


def compute_points(model_data: Dict[str, Any], background_default: float) -> Tuple[np.ndarray, np.ndarray, float, float, str]:
    r = model_data['parameters']['risk_aversion_r']
    beta = model_data['parameters'].get('choice_sensitivity_beta', None)
    model = model_data.get('model', model_data['__file'].stem)
    background = model_data.get('background_wealth', background_default)
    xs, ys = [], []
    for trial in model_data.get('raw_trials', []):
        if 'prob_win' not in trial or 'safe_payoff' not in trial or 'risky_payoff' not in trial:
            continue
        x = expected_utility_diff(trial, r, background)
        if np.isnan(x):
            continue
        y = trial.get('risky_rate')
        if y is None:
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys), r, (beta if beta is not None else np.nan), model


def winsorize(xs: np.ndarray, clip_pct: float) -> np.ndarray:
    if clip_pct <= 0:
        return xs
    lo = np.percentile(xs, clip_pct)
    hi = np.percentile(xs, 100 - clip_pct)
    return np.clip(xs, lo, hi)


def normalize(xs_list: List[np.ndarray], method: str, betas: Optional[List[float]] = None) -> Tuple[List[np.ndarray], List[float]]:
    """Normalize list of arrays.

    Returns (normalized_arrays, scales). For non-linear transforms (rank, logmaxabs)
    the scale is an approximate characteristic scale used only for reporting / logistic mapping.
    For methods that include a shift (zscore, robust, minmax, rank, pooledz) the scale
    returned is the multiplicative component (std, MAD, half-range, etc.).
    """
    scales: List[float] = []

    if method == 'globalmax':
        global_max = max((np.max(np.abs(xs)) for xs in xs_list if xs.size > 0), default=1.0)
        if global_max == 0:
            global_max = 1.0
        scales = [global_max if xs.size > 0 else 1.0 for xs in xs_list]
        return [xs / s if s != 0 else xs for xs, s in zip(xs_list, scales)], scales

    if method == 'pooledz':
        all_concat = np.concatenate([xs for xs in xs_list if xs.size > 0])
        g_mean = np.mean(all_concat)
        g_std = np.std(all_concat)
        if g_std == 0:
            g_std = 1.0
        normed = []
        for xs in xs_list:
            if xs.size == 0:
                normed.append(xs)
                scales.append(1.0)
            else:
                normed.append((xs - g_mean) / g_std)
                scales.append(g_std)
        return normed, scales

    normed: List[np.ndarray] = []
    for idx, xs in enumerate(xs_list):
        if xs.size == 0:
            normed.append(xs)
            scales.append(1.0)
            continue
        if method == 'maxabs':
            s = np.max(np.abs(xs)) or 1.0
            normed.append(xs / s)
            scales.append(s)
        elif method == 'zscore':
            mu = np.mean(xs)
            sd = np.std(xs) or 1.0
            normed.append((xs - mu) / sd)
            scales.append(sd)
        elif method == 'robust':
            med = np.median(xs)
            mad = np.median(np.abs(xs - med))
            if mad == 0:
                alt = np.max(np.abs(xs)) or 1.0
                normed.append(xs / alt)
                scales.append(alt)
            else:
                rob_scale = 1.4826 * mad
                normed.append((xs - med) / rob_scale)
                scales.append(rob_scale)
        elif method == 'logmaxabs':
            m = np.max(np.abs(xs)) or 1.0
            normed.append(np.sign(xs) * np.log1p(np.abs(xs)) / np.log1p(m))
            scales.append(m)
        elif method == 'minmax':
            mn = np.min(xs)
            mx = np.max(xs)
            if mx - mn == 0:
                normed.append(np.zeros_like(xs))
                scales.append(1.0)
            else:
                normed.append(2 * (xs - mn) / (mx - mn) - 1)
                scales.append(0.5 * (mx - mn))  # half-range
        elif method == 'rank':
            order = np.argsort(xs)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(xs.size)
            if xs.size > 1:
                normed.append(2 * (ranks / (xs.size - 1)) - 1)
            else:
                normed.append(np.zeros_like(xs))
            scales.append(np.median(np.abs(xs - np.median(xs))) * 1.4826 or (np.std(xs) or 1.0))
        elif method == 'beta':
            if betas is None or np.isnan(betas[idx]) or betas[idx] == 0:
                normed.append(xs)
                scales.append(1.0)
            else:
                b = betas[idx]
                normed.append(b * xs)  # x' = beta * x  => logistic becomes expit(x')
                scales.append(1.0 / b)  # store inverse for mapping back
        else:
            raise ValueError(f"Unknown norm method: {method}")
    return normed, scales


def main():
    parser = argparse.ArgumentParser(description='Plot normalized EU difference vs risk choice probability across models.')
    parser.add_argument('--analysis-dir', type=Path, required=True, help='Directory containing CRRA JSON files.')
    parser.add_argument('--pattern', type=str, default='crra_analysis_*.json', help='File glob pattern (default crra_analysis_*.json)')
    parser.add_argument('--norm', type=str, default='maxabs', choices=['maxabs', 'zscore', 'globalmax', 'robust', 'logmaxabs', 'minmax', 'rank', 'pooledz', 'beta'], help='Normalization method for x values.')
    parser.add_argument('--background-wealth', type=float, default=100.0, help='Default background wealth if absent in file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG filename.')
    parser.add_argument('--no-curve', action='store_true', help='Do not draw fitted logistic curves, only scatter points.')
    parser.add_argument('--clip-pct', type=float, default=0.0, help='Winsorize EU diffs at this percentile (e.g. 1 => clip 1st/99th).')
    parser.add_argument('--export-csv', type=Path, default=None, help='Path to export per-point data CSV.')
    parser.add_argument('--curve-samples', type=int, default=300, help='Number of samples for logistic curve.')
    parser.add_argument('--raw-x-hist', action='store_true', help='Also create histogram of raw EU differences.')
    args = parser.parse_args()

    files = load_files(args.analysis_dir, args.pattern)
    if not files:
        raise SystemExit('No matching CRRA JSON files found.')

    # Compute raw points
    all_xs, all_ys, rs, betas, labels = [], [], [], [], []
    for md in files:
        xs, ys, r, beta, model = compute_points(md, args.background_wealth)
        if xs.size == 0:
            continue
        all_xs.append(xs)
        all_ys.append(ys)
        rs.append(r)
        betas.append(beta)
        labels.append(model)

    if not all_xs:
        raise SystemExit('No valid trial data found in files.')

    # Optional clipping
    if args.clip_pct > 0:
        all_xs = [winsorize(xs, args.clip_pct) for xs in all_xs]

    # Normalize x
    norm_xs, scales = normalize(all_xs, args.norm, betas=betas)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    export_rows = []
    for i, (x_norm, ys, r, beta, model, orig_x, scale) in enumerate(zip(norm_xs, all_ys, rs, betas, labels, all_xs, scales)):
        color = cmap(i % 10)
        plt.scatter(x_norm, ys, s=35, alpha=0.6, color=color, edgecolor='white', linewidth=0.5, label=f"{model} (r={r:.2f})")
        if args.export_csv:
            for ox, xn, y in zip(orig_x, x_norm, ys):
                export_rows.append({'model': model, 'r': r, 'beta': beta, 'eu_diff_raw': ox, 'eu_diff_norm': xn, 'risky_rate': y})
        # Fitted curve if beta available (note argparse converts --no-curve to no_curve)
        if not args.no_curve and not np.isnan(beta):
            # For some non-linear or rank-based transforms a parametric logistic overlay isn't meaningful.
            skip_curve = args.norm in {'rank'}
            if not skip_curve:
                x_grid_norm = np.linspace(np.min(x_norm), np.max(x_norm), args.curve_samples)
                if args.norm == 'zscore':
                    mu = np.mean(orig_x)
                    sd = np.std(orig_x) if np.std(orig_x) > 0 else 1.0
                    x_grid_orig = mu + sd * x_grid_norm
                elif args.norm == 'logmaxabs':
                    m = scale if scale != 0 else 1.0
                    x_grid_orig = np.sign(x_grid_norm) * (np.expm1(np.abs(x_grid_norm) * np.log1p(m)))
                elif args.norm == 'minmax':
                    mn = np.min(orig_x)
                    rng = np.max(orig_x) - mn
                    if rng == 0:
                        x_grid_orig = np.zeros_like(x_grid_norm) + mn
                    else:
                        x_grid_orig = mn + ((x_grid_norm + 1) / 2) * rng
                elif args.norm == 'beta':
                    # x_norm = beta * x_orig => x_orig = x_norm / beta
                    b = beta if beta != 0 else 1.0
                    x_grid_orig = x_grid_norm / b
                else:
                    # maxabs, globalmax, robust, pooledz use linear scaling stored in 'scale'
                    x_grid_orig = scale * x_grid_norm
                y_fit = expit(beta * x_grid_orig)
                plt.plot(x_grid_norm, y_fit, color=color, linewidth=2)

    plt.axvline(0, color='black', linewidth=1, alpha=0.4, linestyle='--')
    norm_label = {
        'maxabs': '|EU_diff| / max |EU_diff| per model',
        'globalmax': '|EU_diff| / global max |EU_diff|',
        'zscore': '(EU_diff - mean)/std per model',
        'robust': '(EU_diff - median)/(1.4826*MAD)',
        'logmaxabs': 'sign(x)*log(1+|x|)/log(1+max|x|)',
        'minmax': 'Scaled to [-1,1] per model',
        'rank': 'Rank scaled to [-1,1] per model',
        'pooledz': 'Global z-score across models',
        'beta': 'beta * EU_diff (slope-normalized)'
    }[args.norm]
    plt.xlabel(f'Normalized Expected Utility Difference ({norm_label})')
    plt.ylabel('Probability of Choosing Risky')
    plt.title('Normalized CRRA Expected Utility Difference vs Risky Choice Probability')
    plt.legend(fontsize=9, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_file = args.output or (args.analysis_dir / f'crra_eu_diff_{args.norm}_normalized.png')
    plt.savefig(out_file, dpi=180)
    print(f'Saved plot to {out_file}')

    print('\nModels processed:')
    for model, r, beta, scale in zip(labels, rs, betas, scales):
        print(f'  {model:12s} r={r:6.3f} beta={beta if not np.isnan(beta) else float("nan"):.3f} scale={scale:.4g}')

    if args.export_csv and export_rows:
        with open(args.export_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=export_rows[0].keys())
            writer.writeheader()
            writer.writerows(export_rows)
        print(f'Exported data to {args.export_csv}')

    if args.raw_x_hist:
        plt.figure(figsize=(10,6))
        for i,(orig_x, model) in enumerate(zip(all_xs, labels)):
            plt.hist(orig_x, bins=40, alpha=0.45, label=model)
        plt.xlabel('Raw EU Difference (EU_risky - EU_safe)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Raw Expected Utility Differences')
        plt.legend(fontsize=9)
        plt.tight_layout()
        hfile = args.analysis_dir / f'crra_eu_diff_raw_hist.png'
        plt.savefig(hfile, dpi=160)
        plt.close()
        print(f'Saved histogram to {hfile}')


if __name__ == '__main__':
    main()
