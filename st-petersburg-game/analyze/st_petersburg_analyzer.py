"""
St. Petersburg Paradox Analysis and Discount Factor Estimation

This script analyzes experimental results from the St. Petersburg Paradox experiments,
providing comprehensive statistical analysis and behavioral modeling of AI decision-making
under theoretically infinite expected value scenarios.

WHAT THIS SCRIPT DOES:
- Loads and processes St. Petersburg game experimental results from JSON files
- Estimates discount factors (δ) using OLS regression on logarithmic transformations
- Computes auxiliary logistic goodness-of-fit diagnostics
- Generates publication-quality plots and statistical summaries
- Provides behavioral analysis including certainty equivalents and risk profiles

THEORETICAL MODEL:
The analysis fits a discount factor model where the probability of playing decreases
with entry price according to: ln(p) = α - δ ln(X) + ε, where:
- p = probability of playing (0 < p ≤ 1)
- X = entry price
- δ = discount factor (primary parameter of interest)
- α = intercept parameter

This follows the paper-aligned approach for analyzing bounded rationality in
decision-making under theoretically infinite expected value scenarios.

AUXILIARY DIAGNOSTICS:
- Logistic regression for goodness-of-fit assessment
- CRRA (Constant Relative Risk Aversion) parameter estimation
- Certainty equivalent calculations
- Behavioral risk profile classification

REQUIREMENTS:
- Python 3.8+
- Required packages: numpy, pandas, matplotlib, scipy, glob
- St. Petersburg experiment result files in JSON format

USAGE:
1. Run from st-petersburg-game/analyze/ directory
2. Default: python st_petersburg_analyzer.py (searches results/ subdirectories)
3. Results saved as CSV files and publication-ready plots

OUTPUT:
- CSV files with comprehensive fit metrics and play curves
- Summary plot with discount factors, R² values, and play rate curves
- Statistical analysis suitable for academic publication

THEORETICAL SIGNIFICANCE:
The discount factor δ quantifies how sharply willingness-to-pay declines with
price, revealing the degree of bounded rationality in AI decision-making despite
theoretically infinite expected value of the St. Petersburg game.
"""

import glob
import json
import logging
import os
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t

# ============= CONFIGURATION =============
# Modify these values to customize the analysis

# Directory configuration
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "st-petersburg-games" / "results"
# TODO: Set your results directory path here if different from default

# Analysis parameters
DEFAULT_ANALYSIS_TYPE = "persona"  # "neutral" or "persona"
DEFAULT_MAX_ROUNDS = 30  # Maximum rounds for EU truncation
DEFAULT_INITIAL_WEALTH = 1000.0  # Initial wealth for CRRA utility calculations

# Output file names
METRICS_CSV_FILENAME = "stpetersburg_fit_metrics.csv"
CURVES_CSV_FILENAME = "stpetersburg_play_curves.csv"
PLOT_PNG_FILENAME = "stpetersburg_summary.png"

# Plot configuration
PLOT_FIGSIZE = (14, 10)  # Figure size for summary plots
PLOT_DPI = 300  # DPI for saved plots
PLOT_FONTSIZE = 10  # Base font size for plots

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============= END CONFIGURATION =============

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class FitMetrics:
    """
    Container for comprehensive St. Petersburg Paradox analysis results.

    Stores all fitted parameters and diagnostic metrics from the discount factor
    analysis and auxiliary behavioral modeling of AI decision-making.

    Primary Parameters (Paper-Aligned OLS):
    - delta: Discount factor δ quantifying bounded rationality
    - alpha_intercept: Intercept α in ln(p) = α - δ ln(X) + ε
    - r2_ols: Goodness-of-fit for the primary discount model

    Auxiliary Diagnostics (Logistic Regression):
    - log_likelihood, aic, bic: Model comparison metrics
    - pseudo_r2_mcfadden: Pseudo R² for logistic fit
    - rmse, accuracy_50: Prediction accuracy metrics

    Behavioral Analysis:
    - crra_gamma_mle: Estimated CRRA risk aversion parameter
    - certainty_equivalent: Price at which play probability = 50%
    - overall_play_rate: Average proportion of games played
    - risk_profile: Qualitative risk behavior classification
    """

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
# Analyzer Class
# ------------------------------
class StPetersburgAnalyzer:
    """
    Comprehensive analyzer for St. Petersburg Paradox experimental results.

    Provides paper-aligned discount factor estimation using OLS regression on
    logarithmic transformations, along with auxiliary logistic diagnostics and
    behavioral analysis for understanding AI bounded rationality.

    Key Features:
    - Discount factor δ estimation via ln(p) = α - δ ln(X) + ε
    - Logistic regression goodness-of-fit diagnostics
    - CRRA parameter estimation for risk aversion analysis
    - Certainty equivalent calculations
    - Publication-ready plots and statistical summaries
    - Behavioral risk profile classification

    The primary analysis follows the theoretical framework where decision-making
    under theoretically infinite expected value reveals the degree of bounded
    rationality through the discount factor δ.
    """

    def __init__(self) -> None:
        """
        Initialize the St. Petersburg analyzer.

        Sets up data structures for storing experimental results and
        fitted model parameters.
        """
        self.raw_data: List[Dict[str, Any]] = []
        self.results: Dict[str, Dict[str, Any]] = {}

    # ---------- Loading Methods ----------
    def load_json_results(self, results_dir: str) -> None:
        """
        Load experimental results from JSON files in the specified directory.

        Scans the results directory for JSON files, loads each file's data,
        and stores the results for subsequent analysis.

        Args:
            results_dir: Path to directory containing JSON result files

        Raises:
            FileNotFoundError: If the specified directory doesn't exist
            json.JSONDecodeError: If any JSON file is malformed
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        json_pattern = str(results_path / "*.json")
        json_files = sorted(glob.glob(json_pattern))

        logger.info(f"Found {len(json_files)} JSON files in {results_dir}:")
        for fp in json_files:
            logger.info(f"  • {Path(fp).name}")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    self.raw_data.append(json.load(f))
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Failed to load {fp}: {e}")
                raise

        logger.info(f"Successfully loaded {len(self.raw_data)} result files\n")

    # ---------- Parsing Methods ----------
    @staticmethod
    def extract_play_data(
        model_data: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Extract play probabilities and prices from experimental results.

        Parses the experimental data structure to extract entry prices and
        corresponding play rates (probabilities of choosing to play) for
        each price point tested.

        Args:
            model_data: Dictionary containing experimental results with structure:
                - experiment_info.model_tested: AI model name
                - summary_statistics.by_price_breakdown: Dict mapping price strings
                  to statistics including play_percentage

        Returns:
            Tuple of (prices, play_rates, model_name):
            - prices: Sorted numpy array of entry prices (floats)
            - play_rates: Corresponding play probabilities [0,1] (numpy array)
            - model_name: String identifier of the AI model

        Raises:
            KeyError: If expected data structure is missing
            ValueError: If price parsing fails
        """
        try:
            model_name = model_data["experiment_info"]["model_tested"]
            by_price = model_data["summary_statistics"]["by_price_breakdown"]
        except KeyError as e:
            raise KeyError(f"Missing expected data structure: {e}")

        prices: List[float] = []
        play_rates: List[float] = []

        for price_str, stats in by_price.items():
            try:
                # Parse price string (e.g., "$1,000" -> 1000.0)
                price = float(price_str.replace("$", "").replace(",", ""))
                rate = float(stats["play_percentage"]) / 100.0
                prices.append(price)
                play_rates.append(rate)
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse price data for {price_str}: {e}")
                continue

        # Sort by price for consistent analysis
        idx = np.argsort(prices)
        return np.array(prices)[idx], np.array(play_rates)[idx], model_name

    # ---------- Utility Functions ----------
    @staticmethod
    def utility_crra(wealth: float, gamma: float) -> float:
        """
        Compute CRRA (Constant Relative Risk Aversion) utility function.

        U(w) = w^(1-γ)/(1-γ) for γ ≠ 1, U(w) = ln(w) for γ = 1

        Args:
            wealth: Wealth level (must be positive)
            gamma: Relative risk aversion parameter

        Returns:
            CRRA utility value, with safeguards for edge cases
        """
        if wealth <= 0:
            return -1e10  # Large negative penalty for non-positive wealth

        if abs(gamma - 1.0) < 1e-6:
            return float(np.log(max(wealth, 1e-9)))

        return float((wealth ** (1 - gamma)) / (1 - gamma))

    def eu_st_petersburg(
        self,
        entry_price: float,
        gamma: float,
        initial_wealth: float = DEFAULT_INITIAL_WEALTH,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> float:
        """
        Compute expected utility for St. Petersburg game with finite truncation.

        Calculates the expected CRRA utility of playing the St. Petersburg game
        at a given entry price, using finite truncation of the infinite series.
        Used for auxiliary behavioral analysis and CRRA parameter estimation.

        Note: This is not the primary analysis method (paper uses discount factors),
        but provides useful behavioral diagnostics.

        Args:
            entry_price: Cost to enter the game
            gamma: CRRA risk aversion parameter
            initial_wealth: Starting wealth before the game
            max_rounds: Maximum coin flips to consider (truncation point)

        Returns:
            Expected CRRA utility of playing the game
        """
        eu = 0.0
        for n in range(1, max_rounds + 1):
            p = 0.5**n  # Probability of exactly n tails then heads
            payoff = 2 ** (n - 1)  # Payoff after n-1 tails
            final_wealth = initial_wealth - entry_price + payoff
            utility = self.utility_crra(final_wealth, gamma)
            eu += p * utility

        return float(eu)

    # ---------- Paper-Aligned Discount Factor Estimation ----------
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
        Estimate discount factor δ using OLS regression on logarithmic transformation.

        Implements the paper-aligned approach: ln(p_i) = α - δ ln(X_i) + ε_i
        where p_i > 0 (only positive play probabilities are used).

        This method fits the primary behavioral model where the discount factor δ
        quantifies how sharply willingness-to-pay declines with entry price,
        revealing the degree of bounded rationality.

        Args:
            prices: Array of entry prices (X_i values)
            play_rates: Corresponding array of play probabilities (p_i values)

        Returns:
            Tuple of (delta, delta_se, delta_pvalue, alpha, r2, n_used):
            - delta: Estimated discount factor δ (primary parameter)
            - delta_se: Standard error of δ estimate
            - delta_pvalue: P-value for testing δ ≠ 0
            - alpha: Estimated intercept α
            - r2: R-squared goodness-of-fit
            - n_used: Number of data points used (p_i > 0)
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
        Perform comprehensive analysis of all models using discount factor approach.

        Runs the complete analysis pipeline for each model:
        1. Primary OLS discount factor estimation (paper-aligned)
        2. Auxiliary CRRA parameter estimation
        3. Logistic goodness-of-fit diagnostics
        4. Certainty equivalent calculation
        5. Risk profile classification

        The discount factor δ is the primary behavioral parameter, quantifying
        how sharply willingness-to-pay declines with entry price, revealing
        the degree of bounded rationality in AI decision-making.

        Returns:
            pandas.DataFrame: Analysis results with one row per model,
            indexed by model name with comprehensive fit metrics

        Populates:
            self.results: Dictionary of detailed results per model
            self.fit_df_: Summary DataFrame for export
            self.play_curve_df_: Detailed play curves for plotting
        """
        rows: List[FitMetrics] = []
        play_curve_rows: List[Dict[str, Any]] = []

        logger.info("=== ST. PETERSBURG ANALYSIS (paper-aligned) ===\n")

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

            # Log analysis summary for this model
            logger.info(f"📊 {model}:")
            if delta is not None:
                pval_str = f"{delta_pvalue:.3g}" if delta_pvalue is not None else "N/A"
                logger.info(
                    f"   δ (discount factor): {delta:.3f} (α={alpha:.3f}, R²={r2_ols:.3f}, n={n_used}, p={pval_str})"
                )
            else:
                logger.info(
                    "   δ (discount factor): N/A (insufficient positive play rates)"
                )
            if gamma_hat is not None:
                logger.info(f"   γ (CRRA, aux): {gamma_hat:.3f} [{rlabel}]")
            else:
                logger.info("   γ (CRRA, aux): N/A")
            if ce is not None:
                logger.info(f"   Certainty Eq.: ${ce:,.2f}")
            logger.info(f"   Overall Play Rate: {overall_play_rate:.1%}\n")

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

    # ---------- Export Methods ----------
    def export_results(
        self,
        metrics_csv: str = METRICS_CSV_FILENAME,
        curves_csv: str = CURVES_CSV_FILENAME,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export analysis results to CSV files.

        Saves the fitted model metrics and play curves to CSV format for
        further analysis or publication.

        Args:
            metrics_csv: Filename for model metrics CSV
            curves_csv: Filename for play curves CSV

        Returns:
            Tuple of (metrics_df, curves_df): The exported DataFrames

        Raises:
            RuntimeError: If analyze_all_models() hasn't been run yet
        """
        if not hasattr(self, "fit_df_"):
            raise RuntimeError("Run analyze_all_models() before export.")

        # Export to CSV files
        self.fit_df_.to_csv(metrics_csv, index=True)
        self.play_curve_df_.to_csv(curves_csv, index=False)

        logger.info("💾 Exported analysis results:")
        logger.info(f"   📊 Metrics CSV: {Path(metrics_csv).resolve()}")
        logger.info(f"   📈 Curves CSV: {Path(curves_csv).resolve()}")

        return self.fit_df_, self.play_curve_df_

    # ---------- Plotting Methods ----------
    def create_summary_plot(self, save_path: Optional[str] = PLOT_PNG_FILENAME) -> None:
        """
        Create publication-quality summary plot of analysis results.

        Generates a 2x2 subplot figure showing:
        - Discount factor δ estimates across models
        - OLS R² goodness-of-fit
        - Certainty equivalents (50% play probability)
        - Play rate curves vs. entry price (log scale)

        Args:
            save_path: Path to save the plot image (PNG format)

        Raises:
            RuntimeError: If analyze_all_models() hasn't been run yet
        """
        if not hasattr(self, "fit_df_"):
            logger.error("No results to plot. Run analyze_all_models() first.")
            return

        df = self.fit_df_.copy()
        models = list(df.index)

        # Prepare figure with configured size
        fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE)
        ax1, ax2, ax3, ax4 = axes.ravel()

        # Discount factor δ bar plot
        ax1.bar(models, df["delta"].fillna(0.0).values)
        ax1.set_title("Discount Factor δ (OLS, primary)", fontsize=PLOT_FONTSIZE)
        ax1.set_ylabel("δ", fontsize=PLOT_FONTSIZE)
        ax1.tick_params(axis="x", rotation=45)

        # OLS R² bar plot
        ax2.bar(models, df["r2_ols"].fillna(0.0).values)
        ax2.set_title("OLS R² for ln(p) ~ ln(X)", fontsize=PLOT_FONTSIZE)
        ax2.set_ylabel("R²", fontsize=PLOT_FONTSIZE)
        ax2.tick_params(axis="x", rotation=45)

        # Certainty Equivalent bar plot
        ax3.bar(models, df["certainty_equivalent"].fillna(0.0).values)
        ax3.set_title("Certainty Equivalent (50% Play)", fontsize=PLOT_FONTSIZE)
        ax3.set_ylabel("Price ($)", fontsize=PLOT_FONTSIZE)
        ax3.set_yscale("log")
        ax3.tick_params(axis="x", rotation=45)

        # Play curves line plot
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
        ax4.set_title("Play Rate vs. Entry Price", fontsize=PLOT_FONTSIZE)
        ax4.set_xlabel("Entry Price ($)", fontsize=PLOT_FONTSIZE)
        ax4.set_ylabel("Play Rate", fontsize=PLOT_FONTSIZE)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=PLOT_FONTSIZE - 2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
            logger.info(f"📊 Saved summary plot: {Path(save_path).resolve()}")
        plt.close(fig)


# ------------------------------
# CLI Entrypoint
# ------------------------------
def main() -> None:
    """
    Main entry point for St. Petersburg Paradox analysis.

    Automatically discovers experimental results and performs comprehensive
    analysis with discount factor estimation, auxiliary diagnostics, and
    publication-ready outputs.
    """
    # Determine project structure and results directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up one level from 'analyze' to project root

    # Try both neutral and persona subdirectories (configurable)
    neutral_results = project_root / "results" / "neutral"
    persona_results = project_root / "results" / "persona"

    # Choose which results to analyze (configurable via DEFAULT_ANALYSIS_TYPE)
    if DEFAULT_ANALYSIS_TYPE == "persona":
        results_dir = persona_results
        suffix = "_persona"
    else:
        results_dir = neutral_results
        suffix = "_neutral"

    logger.info(f"🎯 St. Petersburg Paradox Analysis")
    logger.info(f"📂 Script directory: {script_dir}")
    logger.info(f"🏠 Project root: {project_root}")
    logger.info(f"📊 Analysis type: {DEFAULT_ANALYSIS_TYPE}")
    logger.info(f"🔍 Results directory: {results_dir}")

    # Check if directory exists
    if not results_dir.exists():
        logger.error(f"❌ Results directory not found: {results_dir}")
        logger.info("📋 Available result directories:")
        results_base = project_root / "results"
        if results_base.exists():
            for item in results_base.iterdir():
                if item.is_dir():
                    logger.info(f"   📁 {item.name}")
        else:
            logger.error(f"❌ Results base directory not found: {results_base}")
        return

    # Configure output filenames
    metrics_csv = f"{METRICS_CSV_FILENAME.replace('.csv', suffix)}.csv"
    curves_csv = f"{CURVES_CSV_FILENAME.replace('.csv', suffix)}.csv"
    plot_png = f"{PLOT_PNG_FILENAME.replace('.png', suffix)}.png"

    logger.info(
        f"📁 Found results directory with {len(list(results_dir.glob('*.json')))} JSON files"
    )

    # Initialize analyzer and load data
    analyzer = StPetersburgAnalyzer()

    try:
        analyzer.load_json_results(str(results_dir))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ Failed to load results: {e}")
        return

    if len(analyzer.raw_data) == 0:
        logger.error("❌ No valid JSON files found in results directory")
        return

    # Perform analysis
    logger.info("🚀 Starting comprehensive analysis...")
    analyzer.analyze_all_models()

    # Export results
    analyzer.export_results(metrics_csv, curves_csv)
    analyzer.create_summary_plot(plot_png)

    logger.info("✅ St. Petersburg Paradox analysis completed successfully!")
    logger.info(f"📈 Results saved to: {Path('.').resolve()}")


if __name__ == "__main__":
    main()
