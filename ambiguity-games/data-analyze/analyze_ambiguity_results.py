from __future__ import annotations
import argparse
import json, math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize


# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "ambiguity-aversion" / "games_outputs" / "ambiguity_game" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Robust path helper ----------------
def _relpath(p):
    try:
        return str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")
@dataclass
class FitResult:
    model: str
    experiment_type: str
    persona_used: bool
    epsilon: float
    beta: float
    r_star: float
    amb_pct: float
    risk_pct: float
    n_choices: int
    source_file: Path
    method: str


# ---------------- Core math ----------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def loglik(theta, xs, successes, trials, ridge=1e-6):
    """Negative log-likelihood + small ridge penalty."""
    eps = 1/(1+np.exp(-theta[0]))  # logistic transform: keep eps in [0,1]
    beta = theta[1]
    VA = (1-eps)*0.5
    VB = xs/100.0
    z = beta*(VA - VB)
    p = sigmoid(z)
    p = np.clip(p, 1e-12, 1-1e-12)
    ll = (successes*np.log(p) + (trials-successes)*np.log(1-p)).sum()
    return -ll + ridge*(beta**2)

def fit_eps_beta(xs, successes, trials):
    xs, successes, trials = map(np.asarray, (xs, successes, trials))
    init = np.array([0.0, 1.0])  # eps≈0.5, beta≈1
    res = optimize.minimize(
        fun=lambda th: loglik(th, xs, successes, trials),
        x0=init, method="L-BFGS-B"
    )
    eps = 1/(1+np.exp(-res.x[0]))
    beta = res.x[1]
    
    # Calculate final NLL for diagnostics
    final_nll = loglik(res.x, xs, successes, trials)
    
    return eps, beta, res.success, final_nll


# ---------------- IO helpers ----------------
def parse_result_file(fp: Path):
    data = json.loads(fp.read_text())
    breakdown = {}
    pct_data = data.get("percentage_breakdown", {})
    for k,v in pct_data.items():
        try:
            red = int(k)
        except Exception:
            try:
                red = int(v.get("red_balls", k))
            except Exception:
                continue  # skip malformed entries
        breakdown[red] = {
            "ambiguous_count": int(v.get("ambiguous_count", 0)),
            "risk_count": int(v.get("risk_count", 0))
        }
    meta = {
        "model": data.get("model","unknown"),
        "experiment_type": data.get("experiment_type","neutral"),
        "persona_used": bool(data.get("persona_used",False))
    }
    records = data.get("all_records", [])
    return breakdown, meta, records

def compute_choice_shares(records):
    n = len(records)
    if n==0: return 0,0,0
    amb = sum(1 for r in records if str(r.get("choice","")).lower()=="ambiguous")
    risk = sum(1 for r in records if str(r.get("choice","")).lower()=="risk")
    return amb/n, risk/n, n

def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


# ---------------- Plotting ----------------
def plot_fits(per_red, eps, beta, model, exp, outfile):
    xs = np.array([row["red_balls"] for row in per_red])
    obs = np.array([row["ambiguous_pct"] for row in per_red])
    VA = (1-eps)*0.5
    fitted = sigmoid(beta*(VA - xs/100))
    plt.figure()
    plt.scatter(xs, obs, label="Observed", color="black")
    plt.plot(xs, fitted, label="Fitted", linewidth=2)
    plt.xlabel("Red balls in risky urn (R)")
    plt.ylabel("P(ambiguous)")
    plt.title(f"{model} ({exp}) fit")
    plt.legend()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()

def render_table_png(df, outfile: Path, title="Ambiguity Game – Summary"):
    plt.figure(figsize=(max(10,0.22*(len(df.columns)+2)*len(df)/6), max(3.5,0.6*len(df))))
    plt.axis('off')
    tbl=plt.table(cellText=df.values,colLabels=df.columns,loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0,1.2)
    plt.title(title,fontsize=12,pad=12)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(outfile,bbox_inches='tight',dpi=200); plt.close()


# ---------------- Processing ----------------
def process_file(fp: Path):
    breakdown, meta, records = parse_result_file(fp)
    xs = sorted(breakdown.keys())
    successes = [breakdown[x]["ambiguous_count"] for x in xs]
    trials = [breakdown[x]["ambiguous_count"]+breakdown[x]["risk_count"] for x in xs]
    
    # Diagnostic logging
    total_trials = sum(trials)
    print(f"Processing: {_relpath(fp)}")
    print(f"  Red levels: {xs}")
    print(f"  Total trials: {total_trials}")
    
    # Check for separation
    for i, (x, succ, n) in enumerate(zip(xs, successes, trials)):
        if succ == 0:
            print(f"  Complete separation at red={x} (0 ambiguous choices)")
        elif succ == n:
            print(f"  Complete separation at red={x} (all ambiguous choices)")
    
    if not xs or total_trials==0: 
        print(f"  SKIPPED: No valid data")
        return None, []

    eps, beta, success, final_nll = fit_eps_beta(xs, successes, trials)
    print(f"  Fit success: {success}, Final NLL: {final_nll:.4f}")
    
    r_star = 50*(1-eps)
    amb_share, risk_share, n_choices = compute_choice_shares(records)

    per_red = []
    VA = (1-eps)*0.5
    for x,succ,n in zip(xs,successes,trials):
        p_hat = sigmoid(beta*(VA - x/100))
        per_red.append({
            "red_balls":x,"ambiguous_count":succ,"risk_count":n-succ,"total":n,
            "ambiguous_pct":succ/n if n else 0,"p_hat":p_hat
        })

    fit = FitResult(
        model=meta["model"], experiment_type=meta["experiment_type"],
        persona_used=meta["persona_used"], epsilon=eps, beta=beta,
        r_star=r_star, amb_pct=amb_share, risk_pct=risk_share,
        n_choices=n_choices, source_file=fp, method="scipy-mle"
    )
    return fit, per_red


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Analyze ambiguity game results")
    parser.add_argument("--root", type=Path, default=RESULTS_ROOT, help="Results root directory")
    parser.add_argument("--pattern", default="ambiguity_results_*.json", help="File pattern to search")
    args = parser.parse_args()
    
    # 1) FILE DISCOVERY (CRITICAL) - Use recursive search
    files = sorted(args.root.rglob(args.pattern))
    
    if not files:
        print(f"No files found. Searched: {args.root} with pattern: {args.pattern}")
        return
    
    print(f"Found {len(files)} files.")
    print("First 10 files:")
    for i, fp in enumerate(files[:10]):
        print(f"  {i+1}: {_relpath(fp)}")
    if len(files) > 10:
        print(f"  ... and {len(files)-10} more")
    print()

    summary_rows = []
    skipped_files = []
    
    for fp in files:
        fit, per_red = process_file(fp)
        if fit is None: 
            skipped_files.append(fp)
            continue
            
        # per-model CSV
        out_subdir = OUT_DIR/"per_model"
        out_subdir.mkdir(parents=True,exist_ok=True)
        per_model_csv = out_subdir/f"per_red_{fit.experiment_type}_{fit.model}.csv"
        write_csv(per_model_csv, per_red, ["red_balls","ambiguous_count","risk_count","total","ambiguous_pct","p_hat"])
        
        # per-model plot
        plot_file = out_subdir/f"fitplot_{fit.experiment_type}_{fit.model}.png"
        plot_fits(per_red, fit.epsilon, fit.beta, fit.model, fit.experiment_type, plot_file)
        
        # Log per-file results
        print(f"  → {fit.model} ({fit.experiment_type}): ε={fit.epsilon:.4f}, β={fit.beta:.4f}, R*={fit.r_star:.3f}")
        print(f"     CSV: {_relpath(per_model_csv)}")
        print(f"     PNG: {_relpath(plot_file)}")
        print()
        
        # 2) SUMMARY APPEND BUG - Ensure this is INSIDE the loop
        summary_rows.append({
            "model":fit.model,"experiment":fit.experiment_type,
            "persona":"yes" if fit.persona_used else "no",
            "epsilon":round(fit.epsilon,4),"beta":round(fit.beta,4),
            "R_star":round(fit.r_star,3),
            "%ambiguous":round(100*fit.amb_pct,1),"%risk":round(100*fit.risk_pct,1),
            "n_choices":fit.n_choices,"method":fit.method,
            "source_file":_relpath(fp)
        })

    # Post-loop validation
    processed_files = len(files) - len(skipped_files)
    if len(summary_rows) != processed_files:
        print(f"WARNING: summary_rows={len(summary_rows)} != processed_files={processed_files}")
        print(f"Skipped files: {[_relpath(f) for f in skipped_files]}")
    
    # 7) SUMMARY CSV
    if not summary_rows:
        print("No valid data found. Skipping summary generation.")
        return
        
    summary_rows.sort(key=lambda r:(r["experiment"],r["model"]))
    summary_csv=OUT_DIR/"sigmoid_fit_summary.csv"
    write_csv(summary_csv, summary_rows,
        ["model","experiment","persona","epsilon","beta","R_star",
         "%ambiguous","%risk","n_choices","method","source_file"])
    print(f"Wrote summary CSV → {_relpath(summary_csv)}")
    
    # 8) ACCEPTANCE CRITERIA
    models = set(r["model"] for r in summary_rows)
    experiments = set(r["experiment"] for r in summary_rows)
    print(f"Summary rows written: {len(summary_rows)}")
    print(f"Distinct models: {len(models)} ({sorted(models)})")
    print(f"Distinct experiments: {len(experiments)} ({sorted(experiments)})")
    
    # Check first 3 per-model files exist
    check_count = min(3, len(summary_rows))
    for i in range(check_count):
        row = summary_rows[i]
        csv_path = OUT_DIR/"per_model"/f"per_red_{row['experiment']}_{row['model']}.csv"
        png_path = OUT_DIR/"per_model"/f"fitplot_{row['experiment']}_{row['model']}.png"
        print(f"File {i+1} check: CSV exists={csv_path.exists()}, PNG exists={png_path.exists()}")

    # 7) SUMMARY PNG
    if summary_rows:
        df=pd.DataFrame(summary_rows)
        fmt_df=df.copy()
        for c in["epsilon","beta","R_star"]: 
            fmt_df[c]=fmt_df[c].map(lambda v:f"{v:.3f}")
        for c in["%ambiguous","%risk"]: 
            fmt_df[c]=fmt_df[c].map(lambda v:f"{v:.1f}")
        fmt_df["n_choices"]=fmt_df["n_choices"].astype(str)
        table_png=OUT_DIR/"sigmoid_fit_summary.png"
        render_table_png(fmt_df, table_png)
        print(f"Wrote table PNG → {_relpath(table_png)}")
    else:
        print("No summary rows to generate PNG from.")


if __name__=="__main__":
    main()
