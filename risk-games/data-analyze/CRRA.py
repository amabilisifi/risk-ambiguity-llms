import json
import math
import os
from typing import List, Tuple, Dict
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ------------------------------------------------------------
# CRRA utility function
# U(w) = (w^(1-r) - 1) / (1 - r) for r != 1, and U(w) = ln(w) for r == 1.
# We subtract 1 in numerator to keep U(w=1)=0 for better comparability.
# This keeps curves comparable across very different r values without
# arbitrary vertical shifts.
# ------------------------------------------------------------

def crra_utility(w: np.ndarray, r: float) -> np.ndarray:
    eps = 1e-12
    w = np.maximum(w, eps)
    if abs(r - 1.0) < 1e-9:
        return np.log(w)
    return (np.power(w, 1.0 - r) - 1.0) / (1.0 - r)


def normalize_curve(u: np.ndarray) -> np.ndarray:
    u_min = np.nanmin(u)
    u_max = np.nanmax(u)
    if not np.isfinite(u_min) or not np.isfinite(u_max) or abs(u_max - u_min) < 1e-12:
        return u
    return (u - u_min) / (u_max - u_min)


def load_model_params(json_path: str) -> Tuple[str, float, float]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model = data.get('model', os.path.basename(json_path))
    ts = data.get('timestamp')
    label = f"{model}" + (f" ({ts})" if ts else '')
    params = data.get('parameters', {})
    r = params.get('risk_aversion_r')
    if r is None:
        raise KeyError(f"risk_aversion_r not found in parameters for {json_path}")
    bg = data.get('background_wealth', 100.0)
    try:
        bg = float(bg)
    except Exception:
        bg = 100.0
    return label, float(r), bg


def strip_parens_text(s: str) -> str:
    """Remove any '(...)' segments from a string."""
    return re.sub(r"\s*\([^)]*\)", "", s).strip()


def plot_crra_set(
    files: List[str],
    title: str,
    out_path: str,
    wealth_min: float = 1.0,
    wealth_max: float = 2000.0,
    n_points: int = 800,
    normalize: bool = True,
    show_background_wealth: bool = True,
):
    plt.figure(figsize=(10, 6))

    # Build wealth grid
    w = np.linspace(wealth_min, wealth_max, n_points)

    all_bgs = []
    missing_files = []

    linestyles = ['-', '--', '-.', ':']
    markers = [None, 'o', None, 's', None, 'd', None, '^', None, 'v', '>', '<', 'x', '+', '*', 'p', 'h']
    linewidth = 2.0

    # Plot each model
    for i, fp in enumerate(files):
        if not os.path.exists(fp):
            missing_files.append(fp)
            continue
        try:
            label, r, bg = load_model_params(fp)
            u = crra_utility(w, r)
            if normalize:
                u = normalize_curve(u)

            display_label = strip_parens_text(label) 
            ls = linestyles[i % len(linestyles)]
            mk = markers[i % len(markers)]

            line, = plt.plot(
                w, u,
                label=f"{display_label} | r={r:.4g}",
                linestyle=ls,
                marker=mk,
                markevery=60,
                linewidth=linewidth,
                alpha=0.95,
                zorder=2,
            )
            line.set_path_effects([pe.Stroke(linewidth=linewidth + 1.5, foreground='white'), pe.Normal()])

            all_bgs.append(bg)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue

    if show_background_wealth and len(all_bgs) > 0:
        bg_unique = np.unique(np.round(all_bgs, 6))
        if len(bg_unique) == 1:
            bw = bg_unique[0]
            if wealth_min <= bw <= wealth_max:
                plt.axvline(bw, color='k', linestyle='--', alpha=0.3, label=f"background wealth = {bw:g}", zorder=1)

    mode = "normalized" if normalize else "raw"
    title_clean = strip_parens_text(title)
    plt.title(f"CRRA Utility - {mode}: {title_clean}")

    plt.xlabel('Wealth')
    plt.ylabel('Utility' + (' (normalized)' if normalize else ''))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    print("sfjkvbkjasbfv j")
    with_character = [
        r"./reults/persona/json/crra_results_gpt-4-1_20250911_121322.json",
        r"./reults/persona/json/crra_results_gpt-4o_20250911_121322.json",
        r"./reults/persona/json/crra_results_gpt-4o-mini_20250911_121322.json",
        r"./reults/persona/json/crra_results_gpt-5_20250911_121321.json",
    ]

    w_o_character = [
        r"./reults/neutral/json/crra_results_gpt-4-1_20250909_151037.json",
        r"./reults/neutral/json/crra_results_gpt-4o_20250909_151036.json",
        r"./reults/neutral/json/crra_results_gpt-4o-mini_20250909_151037.json",
        r"./reults/neutral/json/crra_results_gpt-5_20250909_151036.json",
    ]

    out_dir = r"C:\\Users\\mrgha\\Desktop\\Reserach\\research\\ambiguity-aversion"

    # # Figure 1: prompt_2 only (normalized for comparability)
    # plot_crra_set(
    #     prompt2_files,
    #     title="CRRA Utility (normalized): analysis_prompt_2 models",
    #     out_path=os.path.join(out_dir, "CRRA_prompt2_normalized.png"),
    #     wealth_min=1.0,
    #     wealth_max=2000.0,
    #     n_points=800,
    #     normalize=True,
    # )

    plot_crra_set(
        with_character,
        title="prompt_2 + risk_game models",
        out_path=os.path.join(out_dir, "nCRRA_all_normalized_with_character.png"),
        wealth_min=1.0,
        wealth_max=2000.0,
        n_points=800,
        normalize=True,
    )

    print("ajsjdbvckj")

    plot_crra_set(
        w_o_character,
        title="prompt_2 + risk_game models",
        out_path=os.path.join(out_dir, "nCRRA_all_normalized.png"),
        wealth_min=1.0,
        wealth_max=2000.0,
        n_points=800,
        normalize=True,
    )

    # plot_crra_set(
    #     prompt2_files,
    #     title="CRRA Utility: analysis_prompt_2 models (raw)",
    #     out_path=os.path.join(out_dir, "CRRA_prompt2_raw.png"),
    #     wealth_min=1.0,
    #     wealth_max=2000.0,
    #     n_points=800,
    #     normalize=False,
    # )

    # plot_crra_set(
    #     combined_files,
    #     title="CRRA Utility: prompt_2 + risk_game models (raw)",
    #     out_path=os.path.join(out_dir, "CRRA_all_raw.png"),
    #     wealth_min=1.0,
    #     wealth_max=2000.0,
    #     n_points=800,
    #     normalize=False,
    # )
