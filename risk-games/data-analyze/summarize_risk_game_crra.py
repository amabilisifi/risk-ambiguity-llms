"""Summarize CRRA risk game analysis files into a table (no plots).

Loads files (default pattern: crra_results_*.json) from the specified directory
and computes per-model summary statistics of the Expected Utility (EU) difference
(EU_risky - EU_safe) over trials, along with parameter info.

Outputs:
  - CSV table (default: crra_risk_game_summary.csv in the directory)
  - Optional Markdown table (--md) for quick inclusion in docs/papers.

Columns:
  model, r, beta, trials_with_data, eu_diff_mean, eu_diff_std, eu_diff_median,
  eu_diff_mad, eu_diff_min, eu_diff_max, eu_diff_p1, eu_diff_p99,
  risky_rate_mean, risky_rate_std

Usage (PowerShell):
  python ambiguity-aversion/data-analyze/summarize_risk_game_crra.py \
      --analysis-dir ambiguity-aversion/games_outputs/risk_game/analysis --md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import math
import csv
import numpy as np

# ---------- Utility Functions (CRRA) ----------

def crra_utility(w: float | np.ndarray, r: float) -> float | np.ndarray:
    arr = np.asarray(w)
    arr = np.where(arr <= 0, np.nan, arr)
    if abs(r - 1.0) < 1e-8:
        return np.log(arr)
    return (np.power(arr, 1 - r) - 1) / (1 - r)

def eu_diff(trial: Dict[str, Any], r: float, background: float) -> float:
    try:
        p = trial['prob_win']
        safe = trial['safe_payoff']
        risky = trial['risky_payoff']
    except KeyError:
        return math.nan
    eu_safe = crra_utility(background + safe, r)
    eu_risky_win = crra_utility(background + risky, r)
    eu_risky_lose = crra_utility(background, r)
    eu_risky = p * eu_risky_win + (1 - p) * eu_risky_lose
    return float(eu_risky - eu_safe)

# ---------- Loading & Processing ----------

def load_files(directory: Path, pattern: str) -> List[Dict[str, Any]]:
    out = []
    for f in sorted(directory.glob(pattern)):
        try:
            with f.open('r', encoding='utf-8') as fh:
                content = json.load(fh)
            if 'parameters' not in content or 'risk_aversion_r' not in content['parameters']:
                continue
            content['__file'] = f
            out.append(content)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return out

# ---------- Statistics ----------

def median_abs_deviation(arr: np.ndarray) -> float:
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))

# ---------- Main Summary Logic ----------

def summarize(files: List[Dict[str, Any]], background_default: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for content in files:
        params = content['parameters']
        r = float(params['risk_aversion_r'])
        beta = float(params.get('choice_sensitivity_beta', float('nan')))
        model = content.get('model', content['__file'].stem)
        background = float(content.get('background_wealth', background_default))
        xs = []
        ys = []
        for tr in content.get('raw_trials', []):
            if 'risky_rate' not in tr:
                continue
            x = eu_diff(tr, r, background)
            if not math.isnan(x):
                xs.append(x)
                ys.append(tr['risky_rate'])
        if not xs:
            continue
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        mad = median_abs_deviation(x_arr)
        rows.append({
            'model': model,
            'r': r,
            'beta': beta,
            'trials_with_data': x_arr.size,
            'eu_diff_mean': float(np.mean(x_arr)),
            'eu_diff_std': float(np.std(x_arr)),
            'eu_diff_median': float(np.median(x_arr)),
            'eu_diff_mad': float(mad),
            'eu_diff_min': float(np.min(x_arr)),
            'eu_diff_max': float(np.max(x_arr)),
            'eu_diff_p1': float(np.percentile(x_arr, 1)),
            'eu_diff_p99': float(np.percentile(x_arr, 99)),
            'risky_rate_mean': float(np.mean(y_arr)),
            'risky_rate_std': float(np.std(y_arr)),
        })
    return rows

# ---------- Output Helpers ----------

def write_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        print('No data to write CSV.')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote CSV summary: {path}')

def write_markdown(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        print('No data to write Markdown.')
        return
    headers = list(rows[0].keys())
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f'{v:.4g}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote Markdown table: {path}')

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description='Summarize risk game CRRA result files into a table.')
    p.add_argument('--analysis-dir', type=Path, required=True, help='Directory with CRRA result JSON files.')
    p.add_argument('--pattern', type=str, default='crra_results_*.json', help='Glob pattern (default crra_results_*.json)')
    p.add_argument('--background-wealth', type=float, default=100.0)
    p.add_argument('--csv', type=Path, default=None, help='Output CSV path (default: directory/crra_risk_game_summary.csv)')
    p.add_argument('--md', action='store_true', help='Also write Markdown table.')
    p.add_argument('--md-path', type=Path, default=None, help='Markdown output path (default: directory/crra_risk_game_summary.md)')
    args = p.parse_args()

    files = load_files(args.analysis_dir, args.pattern)
    if not files:
        raise SystemExit('No matching files found.')

    rows = summarize(files, args.background_wealth)
    if not rows:
        raise SystemExit('No usable trial data in files.')

    # Sort by model name
    rows.sort(key=lambda r: r['model'])

    csv_path = args.csv or (args.analysis_dir / 'crra_risk_game_summary.csv')
    write_csv(rows, csv_path)

    if args.md:
        md_path = args.md_path or (args.analysis_dir / 'crra_risk_game_summary.md')
        write_markdown(rows, md_path)

if __name__ == '__main__':
    main()
