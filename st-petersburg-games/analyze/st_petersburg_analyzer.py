# st_petersburg_fit.py
import glob
import json
import os
from dataclasses import asdict, dataclass
from math import log
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t


@dataclass
class FitMetrics:
    model: str
    n_points: int
    # OLS (paper-aligned) discounting fit
    delta: Optional[float]  # discount factor δ (slope sign flipped)
    delta_se: Optional[float]  # robust-ish SE (from simple OLS; see note)
    delta_pvalue: Optional[float]  # p-value for δ (slope)
    alpha_intercept: Optional[float]  # α intercept in ln p = α - δ ln X
    r2_ols: Optional[float]  # R^2 (plain OLS)
    # Logistic fit (auxiliary diagnostics)
    log_likelihood: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    pseudo_r2_mcfadden: Optional[float]
    rmse: Optional[float]
    accuracy_50: Optional[float]
    # Behavioral summaries
    crra_gamma_mle: Optional[float]  # optional (aux) CRRA γ
    certainty_equivalent: Optional[float]
    overall_play_rate: Optional[float]
    risk_profile: Optional[str]


# ------------------------------
# Analyzer
# ------------------------------
class StPetersburgAnalyzer:
    """
    Paper-aligned St. Petersburg fit with discount-factor OLS and
    auxiliary logistic goodness-of-fit diagnostics.
    """

    def __init__(self):
        self.raw_data: List[Dict] = []
        self.results: Dict[str, Dict] = {}

    # ---------- Loading ----------
    def load_json_results(self, results_dir: str) -> None:
        json_files = glob.glob(os.path.join(results_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files:")
        for fp in sorted(json_files):
            print(f"  • {os.path.basename(fp)}")
            with open(fp, "r") as f:
                self.raw_data.append(json.load(f))
        print(f"Loaded {len(self.raw_data)} files.\n")

    # ---------- Parsing ----------
    @staticmethod
    def extract_play_data(model_data: Dict) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Extract prices (float) and play rates ([0,1]) sorted by price.
        Expects structure:
          model_data['experiment_info']['model_tested']
          model_data['summary_statistics']['by_price_breakdown'] = { "$1": {"play_percentage": ...}, ... }
        """
        model_name = model_data["experiment_info"]["model_tested"]
        by_price = model_data["summary_statistics"]["by_price_breakdown"]

        prices, play_rates = [], []
        for price_str, stats in by_price.items():
            price = float(price_str.replace("$", "").replace(",", ""))
            rate = float(stats["play_percentage"]) / 100.0
            prices.append(price)
            play_rates.append(rate)

        idx = np.argsort(prices)
        return np.array(prices)[idx], np.array(play_rates)[idx], model_name

    # ---------- Utilities ----------
    @staticmethod
    def utility_crra(wealth: float, gamma: float) -> float:
        # CRRA utility with safe guards
        if wealth <= 0:
            return -1e10
        if abs(gamma - 1.0) < 1e-6:
            return np.log(max(wealth, 1e-9))
        return (wealth ** (1 - gamma)) / (1 - gamma)

    def eu_st_petersburg(
        self,
        entry_price: float,
        gamma: float,
        initial_wealth: float = 1000.0,
        max_rounds: int = 30,
    ) -> float:
        """
        Expected utility for St. Petersburg (finite truncation).
        Used only for auxiliary logistic diagnostics and γ estimation (not required by paper).
        """
        eu = 0.0
        for n in range(1, max_rounds + 1):
            p = 0.5**n
            payoff = 2 ** (n - 1)
            final_w = initial_wealth - entry_price + payoff
            u = self.utility_crra(final_w, gamma)
            eu += p * u
        return eu

    # ---------- Paper-aligned discounting (OLS on logs) ----------
    def estimate_discount_factor(
        self, prices: np.ndarray, play_rates: np.ndarray
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        int,
    ]:
        """
        Estimate δ from ln(p_i) = α - δ ln(X_i) + ε_i, with p_i > 0 only.
        Returns (delta, delta_se, alpha, r2, n_used).
        """
        # Keep strictly positive probabilities to avoid -inf
        mask = play_rates > 0
        X = prices[mask]
        p = play_rates[mask]

        if X.size < 2:
            return (None, None, None, None, None, int(X.size))

        lnX = np.log(X)
        lnp = np.log(p)

        # OLS via normal equations
        # lnp = b0 + b1*lnX; but b1 should be (-δ). We'll flip sign after fit.
        A = np.vstack([np.ones_like(lnX), lnX]).T
        # beta_hat = (A'A)^{-1} A' y
        ATA = A.T @ A
        try:
            ATA_inv = np.linalg.inv(ATA)
        except np.linalg.LinAlgError:
            ATA_inv = np.linalg.pinv(ATA)
        beta = ATA_inv @ (A.T @ lnp)
        b0, b1 = float(beta[0]), float(beta[1])

        # Residuals and variance
        residuals = lnp - (b0 + b1 * lnX)
        s2 = float((residuals @ residuals) / (len(lnX) - 2)) if len(lnX) > 2 else 0.0
        # Var(b1) = s^2 * (X'X)^{-1}_{11}, where _{11} is the (1,1) index for slope
        var_b1 = s2 * ATA_inv[1, 1] if len(lnX) > 2 else np.nan
        se_b1 = float(np.sqrt(var_b1)) if var_b1 == var_b1 else None  # avoid NaN

        # p-value for slope (b1)
        if se_b1 is not None and se_b1 > 0 and len(lnX) > 2:
            t_stat = b1 / se_b1
            df = len(lnX) - 2
            pval = float(2 * t.sf(abs(t_stat), df))
        else:
            pval = None

        # R^2
        ss_tot = float(((lnp - lnp.mean()) ** 2).sum())
        ss_res = float((residuals**2).sum())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

        # Map to paper notation: α = b0 ; δ = -b1
        alpha = b0
        delta = -b1
        delta_se = se_b1  # SE of b1; SE(δ) = SE(-b1) = SE(b1)

        return delta, delta_se, pval, alpha, r2, int(X.size)

    # ---------- Auxiliary: logistic goodness-of-fit on play curve ----------
    def logistic_fit_metrics(
        self, prices: np.ndarray, play_rates: np.ndarray, gamma_hat: Optional[float]
    ) -> Dict[str, Optional[float]]:
        """
        Given gamma_hat, compute predicted play probs via logistic of EU diff,
        then LL, AIC, BIC, McFadden pseudo-R2, RMSE, and 0.5-accuracy.
        If gamma_hat is None, return Nones for logistic metrics.
        """
        if gamma_hat is None:
            return {
                k: None
                for k in [
                    "log_likelihood",
                    "aic",
                    "bic",
                    "pseudo_r2_mcfadden",
                    "rmse",
                    "accuracy_50",
                ]
            }

        initial_wealth = 1000.0
        # Predicted probabilities
        preds = []
        for price in prices:
            eu_play = self.eu_st_petersburg(price, gamma_hat, initial_wealth)
            eu_hold = self.utility_crra(initial_wealth, gamma_hat)
            diff = (eu_play - eu_hold) / (abs(eu_hold) if eu_hold != 0 else 1.0)
            diff = np.clip(diff, -50, 50)
            preds.append(1.0 / (1.0 + np.exp(-diff)))
        preds = np.array(preds)

        # Log-likelihood against observed aggregated play rates
        eps = 1e-10
        ll = float(
            np.sum(
                play_rates * np.log(preds + eps)
                + (1 - play_rates) * np.log(1 - preds + eps)
            )
        )

        # Null model: constant mean rate
        mean_rate = float(np.mean(play_rates))
        ll_null = float(
            np.sum(
                play_rates * np.log(mean_rate + eps)
                + (1 - play_rates) * np.log(1 - mean_rate + eps)
            )
        )

        # Handle degenerate null likelihood
        if abs(ll_null) < 1e-8:  # or math.isclose(ll_null, 0.0, abs_tol=1e-8)
            pseudo_r2 = None
        else:
            pseudo_r2 = 1 - (ll / ll_null)

        n = len(play_rates)
        k = 1  # only gamma as parameter in this auxiliary model
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        pseudo_r2 = 1 - (ll / ll_null) if ll_null != 0 else None
        rmse = float(np.sqrt(np.mean((preds - play_rates) ** 2)))
        acc = float(np.mean((preds >= 0.5) == (play_rates >= 0.5)))

        return dict(
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            pseudo_r2_mcfadden=pseudo_r2,
            rmse=rmse,
            accuracy_50=acc,
        )

    # ---------- Optional: CRRA γ estimation (auxiliary) ----------
    def estimate_crra_gamma(
        self, prices: np.ndarray, play_rates: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        MLE for a one-parameter CRRA γ using logistic choice based on EU differences.
        Returns (gamma_hat, nll) or (None, None) on failure.
        """

        def nll(g):
            gamma = float(g[0])
            if not (0.01 <= gamma <= 10.0):
                return 1e9
            initial_wealth = 1000.0
            preds = []
            for price in prices:
                eu_play = self.eu_st_petersburg(price, gamma, initial_wealth)
                eu_hold = self.utility_crra(initial_wealth, gamma)
                diff = (eu_play - eu_hold) / (abs(eu_hold) if eu_hold != 0 else 1.0)
                diff = np.clip(diff, -50, 50)
                preds.append(1.0 / (1.0 + np.exp(-diff)))
            preds = np.clip(np.array(preds), 1e-8, 1 - 1e-8)
            ll = np.sum(
                play_rates * np.log(preds) + (1 - play_rates) * np.log(1 - preds)
            )
            return -ll

        best = (None, None)
        for start in [0.5, 1.0, 1.5, 2.0, 3.0]:
            try:
                res = minimize(
                    nll, x0=np.array([start]), bounds=[(0.01, 9.99)], method="L-BFGS-B"
                )
                if res.success:
                    if best[0] is None or res.fun < best[1]:
                        best = (float(res.x[0]), float(res.fun))
            except Exception:
                continue
        return best

    # ---------- Certainty Equivalent ----------
    @staticmethod
    def certainty_equivalent(
        prices: np.ndarray, play_rates: np.ndarray
    ) -> Optional[float]:
        """
        Interpolate CE at 50% play probability along the price curve.
        """
        if len(prices) == 0:
            return None
        for i, r in enumerate(play_rates):
            if r < 0.5:
                if i == 0:
                    return float(prices[0])
                p1, r1 = prices[i - 1], play_rates[i - 1]
                p2, r2 = prices[i], r
                if r2 != r1:
                    w = (0.5 - r1) / (r2 - r1)
                    return float(p1 + w * (p2 - p1))
                return float(p1)
        return float(prices[-1])

    # ---------- Risk profile label ----------
    @staticmethod
    def classify_risk_profile(gamma: Optional[float]) -> str:
        if gamma is None:
            return "Unknown"
        if gamma < 0.5:
            return "Risk Seeking"
        elif gamma < 1.0:
            return "Mildly Risk Averse"
        elif gamma < 2.0:
            return "Moderately Risk Averse"
        elif gamma < 4.0:
            return "Highly Risk Averse"
        else:
            return "Extremely Risk Averse"

    # ---------- Main analysis ----------
    def analyze_all_models(self) -> pd.DataFrame:
        """
        Runs paper-aligned OLS discounting + auxiliary diagnostics for each model.
        Returns a tidy DataFrame (one row per model) and populates self.results.
        """
        rows: List[FitMetrics] = []
        play_curve_rows: List[Dict] = []

        print("=== ST. PETERSBURG ANALYSIS (paper-aligned) ===\n")
        for model_blob in self.raw_data:
            prices, rates, model = self.extract_play_data(model_blob)
            overall_play_rate = (
                float(model_blob["summary_statistics"]["play_percentage"]) / 100.0
            )

            # OLS discount factor (primary, paper-aligned)
            delta, delta_se, delta_pvalue, alpha, r2_ols, n_used = (
                self.estimate_discount_factor(prices, rates)
            )

            # Optional CRRA γ (auxiliary)
            gamma_hat, nll = self.estimate_crra_gamma(prices, rates)

            # Logistic diagnostics (auxiliary, built from γ)
            gof = self.logistic_fit_metrics(prices, rates, gamma_hat)

            # CE
            ce = self.certainty_equivalent(prices, rates)

            # Risk label from γ (aux)
            rlabel = self.classify_risk_profile(gamma_hat)

            # Store per-model structured
            metrics = FitMetrics(
                model=model,
                n_points=int(len(prices)),
                delta=delta,
                delta_se=delta_se,
                delta_pvalue=delta_pvalue,
                alpha_intercept=alpha,
                r2_ols=r2_ols,
                log_likelihood=gof["log_likelihood"],
                aic=gof["aic"],
                bic=gof["bic"],
                pseudo_r2_mcfadden=gof["pseudo_r2_mcfadden"],
                rmse=gof["rmse"],
                accuracy_50=gof["accuracy_50"],
                crra_gamma_mle=gamma_hat,
                certainty_equivalent=ce,
                overall_play_rate=overall_play_rate,
                risk_profile=rlabel,
            )
            rows.append(metrics)

            # Keep long-form play curve for CSV export / plotting
            for p, r in zip(prices, rates):
                play_curve_rows.append(
                    {"Model": model, "Price": float(p), "PlayRate": float(r)}
                )

            # Console summary
            print(f"{model}:")
            if delta is not None:
                pval_str = f"{delta_pvalue:.3g}" if delta_pvalue is not None else "N/A"
                print(
                    f"  δ (discount factor): {delta:.3f}  (α={alpha:.3f}, R²={r2_ols:.3f}, n_used={n_used}, p={pval_str})"
                )
            else:
                print(f"  δ (discount factor): N/A  (insufficient positive play rates)")
            if gamma_hat is not None:
                print(f"  γ (CRRA, aux):      {gamma_hat:.3f}   [{rlabel}]")
            else:
                print(f"  γ (CRRA, aux):      N/A")
            if ce is not None:
                print(f"  Certainty Eq.:      ${ce:,.2f}")
            print(f"  Overall Play Rate:  {overall_play_rate:.1%}\n")

        # Save into self.results
        self.results = {
            row.model: {
                **asdict(row),
                "prices": [
                    d["Price"] for d in play_curve_rows if d["Model"] == row.model
                ],
                "play_rates": [
                    d["PlayRate"] for d in play_curve_rows if d["Model"] == row.model
                ],
            }
            for row in rows
        }

        # Return tidy DF
        fit_df = pd.DataFrame([asdict(r) for r in rows]).set_index("model")
        self.fit_df_ = fit_df
        self.play_curve_df_ = pd.DataFrame(play_curve_rows)
        return fit_df

    # ---------- Export ----------
    def export_results(
        self,
        metrics_csv: str = "stpetersburg_fit_metrics.csv",
        curves_csv: str = "stpetersburg_play_curves.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not hasattr(self, "fit_df_"):
            raise RuntimeError("Run analyze_all_models() before export.")
        self.fit_df_.to_csv(metrics_csv, index=True)
        self.play_curve_df_.to_csv(curves_csv, index=False)
        print(
            f"Saved:\n  • {os.path.abspath(metrics_csv)}\n  • {os.path.abspath(curves_csv)}"
        )
        return self.fit_df_, self.play_curve_df_

    # ---------- Plot ----------
    def create_summary_plot(
        self, save_path: Optional[str] = "stpetersburg_summary.png"
    ) -> None:
        if not hasattr(self, "fit_df_"):
            print("No results to plot. Run analyze_all_models() first.")
            return

        df = self.fit_df_.copy()
        models = list(df.index)

        # Prepare figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axes.ravel()

        # δ bar
        ax1.bar(models, df["delta"].fillna(0.0).values)
        ax1.set_title("Discount Factor δ (OLS, primary)")
        ax1.set_ylabel("δ")
        ax1.tick_params(axis="x", rotation=45)

        # OLS R²
        ax2.bar(models, df["r2_ols"].fillna(0.0).values)
        ax2.set_title("OLS R² for ln(p) ~ ln(X)")
        ax2.set_ylabel("R²")
        ax2.tick_params(axis="x", rotation=45)

        # Certainty Equivalent
        ax3.bar(models, df["certainty_equivalent"].fillna(0.0).values)
        ax3.set_title("Certainty Equivalent (50% Play)")
        ax3.set_ylabel("Price ($, log)")
        ax3.set_yscale("log")
        ax3.tick_params(axis="x", rotation=45)

        # Play curves
        for m in models:
            sub = self.play_curve_df_[self.play_curve_df_["Model"] == m]
            ax4.plot(
                sub["Price"].values,
                sub["PlayRate"].values,
                marker="o",
                linewidth=2,
                label=m,
            )
        ax4.set_xscale("log")
        ax4.set_ylim(-0.02, 1.02)
        ax4.set_title("Play Rate vs. Entry Price")
        ax4.set_xlabel("Entry Price ($, log)")
        ax4.set_ylabel("Play Rate")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {os.path.abspath(save_path)}")
        plt.close(fig)


# ------------------------------
# CLI Entrypoint
# ------------------------------
def main():

    # Get the current script directory and find results relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        script_dir
    )  # Go up one level from 'analyze' to 'st.petersburg-paradox'

    # Try both neutral and persona subdirectories
    neutral_results = os.path.join(project_root, "results", "neutral")
    persona_results = os.path.join(project_root, "results", "persona")

    # Choose which results to analyze (you can modify this)
    RESULTS_DIR = (
        persona_results  # Change to persona_results if you want persona analysis
    )
    print(f"DEBUG: Set RESULTS_DIR to: {RESULTS_DIR}")

    METRICS_CSV = "stpetersburg_fit_metrics_persona.csv"
    CURVES_CSV = "stpetersburg_play_curves_persona.csv"
    PLOT_PNG = "stpetersburg_summary_persona.png"

    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Looking for JSON in: {os.path.abspath(RESULTS_DIR)}")

    # Check if directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory {RESULTS_DIR} does not exist!")
        print("Available directories:")
        if os.path.exists(os.path.join(project_root, "results")):
            for item in os.listdir(os.path.join(project_root, "results")):
                print(f"  - {item}")
        return

    print("Files in results directory:", os.listdir(RESULTS_DIR))

    analyzer = StPetersburgAnalyzer()
    analyzer.load_json_results(RESULTS_DIR)

    if len(analyzer.raw_data) == 0:
        print("No JSON files found. Check the directory path.")
        return

    analyzer.analyze_all_models()
    analyzer.export_results(METRICS_CSV, CURVES_CSV)
    analyzer.create_summary_plot(PLOT_PNG)


if __name__ == "__main__":
    main()
