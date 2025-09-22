"""Plot normalized risk premium (utility difference proxy) vs empirical probability for multiple normalization methods.

Context:
  The provided JSON files in the risk game source_data folder (e.g. gpt-4-1_risk_game_results.json)
  contain a list of objects each with a single key:
      { "risk_premium": <float> }

Goal:
  We treat each risk_premium value as an (unscaled) utility difference proxy. Because the original
  detailed per-trial structure (prob_win, safe_payoff, risky_payoff, risky_rate) is not present
  in these simpler aggregates, we approximate a probability axis using the empirical CDF within
  each model file: for sorted values x_(i), assign p_i = rank(i)/(n-1). This creates a
  monotonic mapping from utility difference (risk premium) to an empirical probability that the
  utility difference is less than or equal to a point.

  We then apply several normalization methods to the risk premium arrays (per model) and generate
  a multi‑panel figure (rows x cols) similar in style to plot_crra_eu_diff_all_norms.py.

Normalization methods implemented (same semantics as existing EU diff scripts where applicable):
    maxabs      : x' = x / max |x|
    globalmax   : x' = x / (global max |x| across all models)
    zscore      : (x - mean)/std per model
    robust      : (x - median)/(1.4826*MAD) fallback to maxabs if MAD=0
    logmaxabs   : sign(x)*log(1+|x|)/log(1+max|x|)
    minmax      : 2*(x - min)/(max-min) - 1 per model
    rank        : Replace x by rank scaled to [-1,1] (order only)
    pooledz     : Global (across models) z-score

Because we lack a parametric choice sensitivity (beta), we only present scatter / line (CDF) plots:
  - Scatter: normalized value vs model-specific empirical probability p_i
  - Optional thin line connecting sorted points to visualize shape.

Usage (PowerShell example):
  python ambiguity-aversion/data-analyze/plot_risk_premium_diff_probability_all_norms.py ^
      --input-dir ambiguity-aversion/games_outputs/risk_game/analysis/source_data ^
      --pattern *_risk_game_results.json ^
      --methods maxabs globalmax zscore robust logmaxabs minmax rank pooledz ^
      --clip-pct 1

Outputs:
  risk_premium_all_normalizations.png (unless --output specified)
  Optional CSV export of the normalized points (--export-csv)

Limitations:
  The empirical probability axis is an approximation (ECDF) and NOT the original probability of
  choosing the risky option. For comparative / shape visualization only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

METHOD_LABEL = {
    'maxabs': 'maxabs',
    'globalmax': 'globalmax',
    'zscore': 'z-score',
    'robust': 'robust',
    'logmaxabs': 'log-maxabs',
    'minmax': 'min-max',
    'rank': 'rank',
    'pooledz': 'pooled z'
}


def load_risk_premium_files(directory: Path, pattern: str) -> List[Dict[str, Any]]:
    data = []
    for f in sorted(directory.glob(pattern)):
        try:
            with f.open('r', encoding='utf-8') as fh:
                content = json.load(fh)
            # Expect a list of objects with 'risk_premium'
            if not isinstance(content, list):
                continue
            arr = []
            for entry in content:
                if isinstance(entry, dict) and 'risk_premium' in entry:
                    val = entry['risk_premium']
                    if val is not None:
                        arr.append(float(val))
            if len(arr) == 0:
                continue
            data.append({'__file': f, 'model': f.stem.replace('_risk_game_results',''), 'risk_premiums': np.array(arr, dtype=float)})
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return data


def winsorize(xs: np.ndarray, clip_pct: float) -> np.ndarray:
    if clip_pct <= 0 or xs.size == 0:
        return xs
    lo = np.percentile(xs, clip_pct)
    hi = np.percentile(xs, 100 - clip_pct)
    return np.clip(xs, lo, hi)


def normalize(xs_list: List[np.ndarray], method: str) -> Tuple[List[np.ndarray], List[float]]:
    scales: List[float] = []
    if method == 'globalmax':
        gmax = max((np.max(np.abs(xs)) for xs in xs_list if xs.size > 0), default=1.0)
        if gmax == 0:
            gmax = 1.0
        out = [(xs / gmax) if gmax != 0 else xs for xs in xs_list]
        scales = [gmax for _ in xs_list]
        return out, scales
    if method == 'pooledz':
        all_concat = np.concatenate([xs for xs in xs_list if xs.size > 0])
        m = np.mean(all_concat)
        s = np.std(all_concat) or 1.0
        out = [ ((xs - m)/s) if xs.size > 0 else xs for xs in xs_list ]
        scales = [s if xs.size > 0 else 1.0 for xs in xs_list]
        return out, scales
    out: List[np.ndarray] = []
    for xs in xs_list:
        if xs.size == 0:
            out.append(xs); scales.append(1.0); continue
        if method == 'maxabs':
            s = np.max(np.abs(xs)) or 1.0
            out.append(xs / s); scales.append(s)
        elif method == 'zscore':
            m = np.mean(xs); s = np.std(xs) or 1.0
            out.append((xs - m) / s); scales.append(s)
        elif method == 'robust':
            med = np.median(xs); mad = np.median(np.abs(xs - med))
            if mad == 0:
                alt = np.max(np.abs(xs)) or 1.0
                out.append(xs / alt); scales.append(alt)
            else:
                rob = 1.4826 * mad
                out.append((xs - med)/rob); scales.append(rob)
        elif method == 'logmaxabs':
            m = np.max(np.abs(xs)) or 1.0
            out.append(np.sign(xs) * np.log1p(np.abs(xs))/np.log1p(m)); scales.append(m)
        elif method == 'minmax':
            mn = np.min(xs); mx = np.max(xs)
            if mx - mn == 0:
                out.append(np.zeros_like(xs)); scales.append(1.0)
            else:
                out.append(2*(xs - mn)/(mx - mn) - 1); scales.append(0.5*(mx - mn))
        elif method == 'rank':
            order = np.argsort(xs)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(xs.size)
            if xs.size > 1:
                out.append(2*(ranks/(xs.size - 1)) - 1)
            else:
                out.append(np.zeros_like(xs))
            # characteristic scale
            scales.append(np.median(np.abs(xs - np.median(xs))) * 1.4826 or (np.std(xs) or 1.0))
        else:
            raise ValueError(f"Unknown method {method}")
    return out, scales


def build_empirical_probabilities(xs: np.ndarray) -> np.ndarray:
    # Empirical CDF probabilities aligned with original order for scatter pairing
    order = np.argsort(xs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(xs.size)
    if xs.size == 1:
        return np.array([0.5])
    return ranks / (xs.size - 1)


def main():
    parser = argparse.ArgumentParser(description='Multi-normalization scatter plots of risk premium vs empirical probability.')
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing *_risk_game_results.json files.')
    parser.add_argument('--pattern', type=str, default='*_risk_game_results.json')
    parser.add_argument('--methods', nargs='*', default=['maxabs','globalmax','zscore','robust','logmaxabs','minmax','rank','pooledz'])
    parser.add_argument('--clip-pct', type=float, default=0.0, help='Winsorize each model series at given percentile (e.g. 1)')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG filename (default risk_premium_all_normalizations.png)')
    parser.add_argument('--no-lines', action='store_true', help='Skip drawing connecting lines through sorted points.')
    parser.add_argument('--export-csv', type=Path, default=None, help='Optional CSV export of normalized points.')
    args = parser.parse_args()

    datasets = load_risk_premium_files(args.input_dir, args.pattern)
    if not datasets:
        raise SystemExit('No matching risk premium files found.')

    # Extract arrays & optionally clip
    raw_arrays = []
    labels = []
    for d in datasets:
        arr = d['risk_premiums']
        if args.clip_pct > 0:
            arr = winsorize(arr, args.clip_pct)
        raw_arrays.append(arr)
        labels.append(d['model'])

    methods = args.methods
    n = len(methods)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.2*rows), sharey=True, sharex=False)
    axes = np.array(axes).reshape(rows, cols)
    cmap = plt.get_cmap('tab10')

    for idx, method in enumerate(methods):
        ax = axes[idx // cols][idx % cols]
        try:
            norm_arrays, scales = normalize(raw_arrays, method)
        except Exception as e:
            ax.text(0.5,0.5,f'Error: {e}', ha='center', va='center'); continue
        for i,(arr_norm, arr_raw, label) in enumerate(zip(norm_arrays, raw_arrays, labels)):
            color = cmap(i % 10)
            probs = build_empirical_probabilities(arr_raw)
            ax.scatter(arr_norm, probs, s=22, alpha=0.6, color=color, edgecolor='white', linewidth=0.4, label=label if idx==0 else None)
            if not args.no_lines:
                # Draw smooth-ish line by sorting normalized x while using corresponding probs
                order = np.argsort(arr_norm)
                ax.plot(arr_norm[order], probs[order], color=color, linewidth=0.9, alpha=0.7)
        ax.set_title(METHOD_LABEL.get(method, method))
        ax.set_xlabel('Normalized risk premium')
        if idx % cols == 0:
            ax.set_ylabel('Empirical probability (CDF)')
        ax.grid(alpha=0.25)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

    # Turn off unused axes
    for j in range(n, rows*cols):
        axes[j // cols][j % cols].axis('off')

    if labels:
        handles = []
        from matplotlib.lines import Line2D
        for i,label in enumerate(labels):
            handles.append(Line2D([0],[0], marker='o', linestyle='None', color=cmap(i%10), label=label, markeredgecolor='white'))
        fig.legend(handles=handles, loc='upper center', ncol=min(len(labels),5), frameon=False, bbox_to_anchor=(0.5, 0.995))

    fig.suptitle('Risk Premium Normalization Comparison (vs Empirical CDF)', fontsize=16, y=0.995)
    fig.tight_layout(rect=(0,0,1,0.965))

    out = args.output or (args.input_dir / 'risk_premium_all_normalizations.png')
    fig.savefig(out, dpi=170)
    print(f'Saved figure to {out}')

    if args.export_csv:
        import csv
        rows_out = []
        for method in methods:
            norm_arrays,_ = normalize(raw_arrays, method)
            for label, arr_raw, arr_norm in zip(labels, raw_arrays, norm_arrays):
                probs = build_empirical_probabilities(arr_raw)
                for raw_v, norm_v, p in zip(arr_raw, arr_norm, probs):
                    rows_out.append({'model': label, 'method': method, 'risk_premium_raw': raw_v, 'risk_premium_norm': norm_v, 'empirical_prob': p})
        with open(args.export_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
            writer.writeheader()
            writer.writerows(rows_out)
        print(f'Exported CSV to {args.export_csv}')

if __name__ == '__main__':
    main()
