"""Create a multi-panel figure comparing all normalization methods for CRRA EU differences.

Generates a 3x3 grid (up to 9 methods) of scatter plots of normalized EU_risky - EU_safe
vs probability of choosing risky for each model file.

Methods included:
  maxabs, globalmax, zscore, robust, logmaxabs, minmax, rank, pooledz, beta

Usage (PowerShell example):
  python ambiguity-aversion/data-analyze/plot_crra_eu_diff_all_norms.py \
      --analysis-dir ambiguity-aversion/games_outputs/risk_game/opportunity_hunter_analysis \
      --pattern crra_analysis_*.json --clip-pct 1

Options mirror the single-normalization script where possible.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# ---------------- Core utility functions (adapted from plot_crra_eu_diff_normalized) ----------------

def crra_utility(w: float | np.ndarray, r: float) -> float | np.ndarray:
    w_arr = np.asarray(w)
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
    beta = model_data['parameters'].get('choice_sensitivity_beta', np.nan)
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
    return np.array(xs), np.array(ys), r, beta, model

def winsorize(xs: np.ndarray, clip_pct: float) -> np.ndarray:
    if clip_pct <= 0:
        return xs
    lo = np.percentile(xs, clip_pct)
    hi = np.percentile(xs, 100 - clip_pct)
    return np.clip(xs, lo, hi)

def normalize(xs_list: List[np.ndarray], method: str, betas: Optional[List[float]] = None) -> Tuple[List[np.ndarray], List[float]]:
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
        g_std = np.std(all_concat) or 1.0
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
            normed.append(xs); scales.append(1.0); continue
        if method == 'maxabs':
            s = np.max(np.abs(xs)) or 1.0
            normed.append(xs / s); scales.append(s)
        elif method == 'zscore':
            mu = np.mean(xs); sd = np.std(xs) or 1.0
            normed.append((xs - mu) / sd); scales.append(sd)
        elif method == 'robust':
            med = np.median(xs); mad = np.median(np.abs(xs - med))
            if mad == 0:
                alt = np.max(np.abs(xs)) or 1.0
                normed.append(xs / alt); scales.append(alt)
            else:
                rob = 1.4826 * mad
                normed.append((xs - med) / rob); scales.append(rob)
        elif method == 'logmaxabs':
            m = np.max(np.abs(xs)) or 1.0
            normed.append(np.sign(xs) * np.log1p(np.abs(xs)) / np.log1p(m)); scales.append(m)
        elif method == 'minmax':
            mn = np.min(xs); mx = np.max(xs)
            if mx - mn == 0:
                normed.append(np.zeros_like(xs)); scales.append(1.0)
            else:
                normed.append(2 * (xs - mn)/(mx - mn) - 1); scales.append(0.5*(mx - mn))
        elif method == 'rank':
            order = np.argsort(xs); ranks = np.empty_like(order); ranks[order] = np.arange(xs.size)
            if xs.size > 1:
                normed.append(2 * (ranks /(xs.size - 1)) - 1)
            else:
                normed.append(np.zeros_like(xs))
            scales.append(np.median(np.abs(xs - np.median(xs))) * 1.4826 or (np.std(xs) or 1.0))
        elif method == 'beta':
            if betas is None or np.isnan(betas[idx]) or betas[idx] == 0:
                normed.append(xs); scales.append(1.0)
            else:
                b = betas[idx]
                normed.append(b * xs)  # logistic becomes expit(x')
                scales.append(1.0 / b)
        else:
            raise ValueError(method)
    return normed, scales

METHOD_LABEL = {
    'maxabs': 'maxabs',
    'globalmax': 'globalmax',
    'zscore': 'z-score',
    'robust': 'robust',
    'logmaxabs': 'log-maxabs',
    'minmax': 'min-max',
    'rank': 'rank',
    'pooledz': 'pooled z',
    'beta': 'beta-scaled'
}

# ---------------- Main multi-panel plotting ----------------

def main():
    parser = argparse.ArgumentParser(description='Multi-panel comparison of normalization methods for CRRA EU differences.')
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--pattern', type=str, default='crra_analysis_*.json')
    parser.add_argument('--background-wealth', type=float, default=100.0)
    parser.add_argument('--clip-pct', type=float, default=0.0, help='Winsorize percentile (e.g. 1) before normalization.')
    parser.add_argument('--methods', nargs='*', default=['maxabs','globalmax','zscore','robust','logmaxabs','minmax','rank','pooledz','beta'],
                        help='Subset of methods to include (default all).')
    parser.add_argument('--curve-samples', type=int, default=200)
    parser.add_argument('--no-curve', action='store_true', help='Skip logistic overlays.')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    files = load_files(args.analysis_dir, args.pattern)
    if not files:
        raise SystemExit('No matching files.')

    raw_xs, raw_ys, rs, betas, labels = [], [], [], [], []
    for md in files:
        xs, ys, r, beta, model = compute_points(md, args.background_wealth)
        if xs.size == 0: continue
        if args.clip_pct > 0:
            xs = winsorize(xs, args.clip_pct)
        raw_xs.append(xs)
        raw_ys.append(ys)
        rs.append(r); betas.append(beta); labels.append(model)

    if not raw_xs:
        raise SystemExit('No valid trial data.')

    methods = args.methods
    n = len(methods)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows), sharey=True)
    axes = np.array(axes).reshape(rows, cols)
    cmap = plt.get_cmap('tab10')

    for idx, method in enumerate(methods):
        ax = axes[idx // cols][idx % cols]
        try:
            norm_xs, scales = normalize(raw_xs, method, betas=betas)
        except Exception as e:
            ax.text(0.5,0.5,f'Error: {e}', ha='center', va='center'); continue
        for i,(x_norm, ys, r, beta, model, orig_x, scale) in enumerate(zip(norm_xs, raw_ys, rs, betas, labels, raw_xs, scales)):
            color = cmap(i % 10)
            ax.scatter(x_norm, ys, s=25, alpha=0.55, color=color, edgecolor='white', linewidth=0.4)
            if not args.no_curve and not np.isnan(beta) and method not in {'rank'}:
                xg = np.linspace(np.min(x_norm), np.max(x_norm), args.curve_samples)
                if method == 'zscore':
                    mu = np.mean(orig_x); sd = np.std(orig_x) or 1.0
                    orig_grid = mu + sd * xg
                elif method == 'logmaxabs':
                    m = scale if scale != 0 else 1.0
                    orig_grid = np.sign(xg) * (np.expm1(np.abs(xg) * np.log1p(m)))
                elif method == 'minmax':
                    mn = np.min(orig_x); rng = np.max(orig_x) - mn
                    orig_grid = mn + ((xg + 1)/2) * (rng if rng != 0 else 0)
                elif method == 'beta':
                    b = beta if beta != 0 else 1.0
                    orig_grid = xg / b  # since x_norm = b * x_orig
                else:
                    orig_grid = scale * xg  # linear
                if method == 'beta':
                    # logistic(beta*orig)= logistic(x_norm)
                    y_fit = expit(xg)
                else:
                    y_fit = expit(beta * orig_grid)
                ax.plot(xg, y_fit, color=color, linewidth=1.4)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.4, linestyle='--')
        ax.set_title(METHOD_LABEL.get(method, method))
        if idx // cols == rows - 1:
            ax.set_xlabel('Normalized EU diff')
        if idx % cols == 0:
            ax.set_ylabel('P(risky)')
        ax.grid(alpha=0.25)

    # Remove unused subplots
    for j in range(n, rows*cols):
        axes[j // cols][j % cols].axis('off')

    # Single legend outside
    handles = []
    from matplotlib.lines import Line2D
    for i,(model,r) in enumerate(zip(labels, rs)):
        handles.append(Line2D([0],[0], marker='o', color='w', label=f'{model} (r={r:.2f})',
                              markerfacecolor=cmap(i % 10), markersize=8, markeredgecolor='white'))
    fig.legend(handles=handles, loc='upper center', ncol=min(len(labels),5), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle('CRRA Expected Utility Difference Normalization Comparison', fontsize=16, y=0.995)
    fig.tight_layout(rect=(0,0,1,0.965))

    out = args.output or (args.analysis_dir / 'crra_eu_diff_all_normalizations.png')
    fig.savefig(out, dpi=170)
    print(f'Saved multi-normalization figure to {out}')

if __name__ == '__main__':
    main()
