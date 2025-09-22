"""Plot normalized risk premium vs empirical probability for a single normalization method.

This is a companion to plot_risk_premium_diff_probability_all_norms.py but focuses on one method
for a clearer combined scatter (all models together) with legend and optional CSV export.

Empirical probability is rank/(n-1) within each model file (ECDF). See the multi-panel script
for methodological caveats.

Usage (PowerShell):
  python ambiguity-aversion/data-analyze/plot_risk_premium_diff_probability_single_norm.py ^
      --input-dir ambiguity-aversion/games_outputs/risk_game/analysis/source_data ^
      --pattern *_risk_game_results.json --norm zscore --clip-pct 1
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

METHOD_CHOICES = ['maxabs','globalmax','zscore','robust','logmaxabs','minmax','rank','pooledz']


def load_risk_premium_files(directory: Path, pattern: str):
    data = []
    for f in sorted(directory.glob(pattern)):
        try:
            content = json.load(f.open('r', encoding='utf-8'))
            if not isinstance(content, list):
                continue
            vals = []
            for entry in content:
                if isinstance(entry, dict) and 'risk_premium' in entry:
                    v = entry['risk_premium']
                    if v is not None:
                        vals.append(float(v))
            if vals:
                data.append({'__file': f, 'model': f.stem.replace('_risk_game_results',''), 'risk_premiums': np.array(vals, dtype=float)})
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return data


def winsorize(xs: np.ndarray, clip_pct: float) -> np.ndarray:
    if clip_pct <= 0 or xs.size == 0:
        return xs
    lo = np.percentile(xs, clip_pct)
    hi = np.percentile(xs, 100 - clip_pct)
    return np.clip(xs, lo, hi)


def normalize(xs_list: List[np.ndarray], method: str):
    if method == 'globalmax':
        gmax = max((np.max(np.abs(xs)) for xs in xs_list if xs.size>0), default=1.0)
        if gmax == 0: gmax = 1.0
        return [xs/gmax for xs in xs_list], [gmax]*len(xs_list)
    if method == 'pooledz':
        allc = np.concatenate([xs for xs in xs_list if xs.size>0])
        m = np.mean(allc); s = np.std(allc) or 1.0
        return [ (xs - m)/s for xs in xs_list ], [s]*len(xs_list)
    outs = []; scales=[]
    for xs in xs_list:
        if xs.size==0:
            outs.append(xs); scales.append(1.0); continue
        if method=='maxabs':
            s = np.max(np.abs(xs)) or 1.0; outs.append(xs/s); scales.append(s)
        elif method=='zscore':
            m=np.mean(xs); s=np.std(xs) or 1.0; outs.append((xs-m)/s); scales.append(s)
        elif method=='robust':
            med=np.median(xs); mad=np.median(np.abs(xs-med))
            if mad==0:
                alt=np.max(np.abs(xs)) or 1.0; outs.append(xs/alt); scales.append(alt)
            else:
                rob=1.4826*mad; outs.append((xs-med)/rob); scales.append(rob)
        elif method=='logmaxabs':
            m=np.max(np.abs(xs)) or 1.0; outs.append(np.sign(xs)*np.log1p(np.abs(xs))/np.log1p(m)); scales.append(m)
        elif method=='minmax':
            mn=np.min(xs); mx=np.max(xs)
            if mx-mn==0: outs.append(np.zeros_like(xs)); scales.append(1.0)
            else: outs.append(2*(xs-mn)/(mx-mn)-1); scales.append(0.5*(mx-mn))
        elif method=='rank':
            order=np.argsort(xs); ranks=np.empty_like(order); ranks[order]=np.arange(xs.size)
            outs.append(2*(ranks/(xs.size-1))-1 if xs.size>1 else np.zeros_like(xs));
            scales.append(np.median(np.abs(xs-np.median(xs)))*1.4826 or (np.std(xs) or 1.0))
        else:
            raise ValueError(method)
    return outs, scales


def empirical_probs(xs: np.ndarray) -> np.ndarray:
    order = np.argsort(xs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(xs.size)
    if xs.size==1: return np.array([0.5])
    return ranks / (xs.size - 1)


def main():
    parser = argparse.ArgumentParser(description='Single normalization risk premium vs empirical probability.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--pattern', type=str, default='*_risk_game_results.json')
    parser.add_argument('--norm', type=str, choices=METHOD_CHOICES, default='maxabs')
    parser.add_argument('--clip-pct', type=float, default=0.0)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--export-csv', type=Path, default=None)
    args = parser.parse_args()

    datasets = load_risk_premium_files(args.input_dir, args.pattern)
    if not datasets:
        raise SystemExit('No matching files')

    raw_arrays = []
    labels = []
    for d in datasets:
        arr = d['risk_premiums']
        if args.clip_pct > 0:
            arr = winsorize(arr, args.clip_pct)
        raw_arrays.append(arr)
        labels.append(d['model'])

    norm_arrays, scales = normalize(raw_arrays, args.norm)

    plt.figure(figsize=(10,6))
    cmap = plt.get_cmap('tab10')
    export_rows = []
    for i,(arr_norm, arr_raw, label) in enumerate(zip(norm_arrays, raw_arrays, labels)):
        color = cmap(i % 10)
        probs = empirical_probs(arr_raw)
        plt.scatter(arr_norm, probs, s=30, alpha=0.65, color=color, edgecolor='white', linewidth=0.5, label=label)
        order = np.argsort(arr_norm)
        plt.plot(arr_norm[order], probs[order], color=color, linewidth=1.2, alpha=0.75)
        if args.export_csv:
            for raw_v, norm_v, p in zip(arr_raw, arr_norm, probs):
                export_rows.append({'model': label, 'norm': args.norm, 'risk_premium_raw': raw_v, 'risk_premium_norm': norm_v, 'empirical_prob': p})

    norm_desc = {
        'maxabs': 'x / max|x| per model',
        'globalmax': 'x / global max|x|',
        'zscore': '(x - mean)/std per model',
        'robust': '(x - median)/(1.4826*MAD)',
        'logmaxabs': 'sign(x)*log(1+|x|)/log(1+max|x|)',
        'minmax': 'Scaled to [-1,1] per model',
        'rank': 'Rank scaled to [-1,1]',
        'pooledz': 'Global z-score across models'
    }[args.norm]

    plt.xlabel(f'Normalized risk premium ({norm_desc})')
    plt.ylabel('Empirical probability (CDF)')
    plt.title('Risk Premium vs Empirical Probability (single normalization)')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = args.output or (args.input_dir / f'risk_premium_{args.norm}_normalized.png')
    plt.savefig(out, dpi=180)
    print(f'Saved plot to {out}')

    if args.export_csv and export_rows:
        import csv
        with open(args.export_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=export_rows[0].keys())
            writer.writeheader()
            writer.writerows(export_rows)
        print(f'Exported CSV to {args.export_csv}')

if __name__ == '__main__':
    main()
