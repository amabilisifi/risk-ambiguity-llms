import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class StPetersburgAnalyzer:
    def __init__(self):
        self.results = {}
        self.raw_data = []

    def load_json_results(self, results_dir: str) -> None:
        """Load all JSON result files from directory"""
        json_files = glob.glob(os.path.join(results_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files to analyze:")

        for file_path in json_files:
            print(f"  - {os.path.basename(file_path)}")
            with open(file_path, "r") as f:
                data = json.load(f)
                self.raw_data.append(data)

        print(f"Successfully loaded {len(self.raw_data)} result files\n")

    def extract_play_data(self, model_data: Dict) -> Tuple[np.array, np.array, str]:
        """Extract prices and play percentages from JSON structure"""
        model_name = model_data["experiment_info"]["model_tested"]
        by_price = model_data["summary_statistics"]["by_price_breakdown"]

        prices = []
        play_percentages = []

        for price_str, stats in by_price.items():
            # Clean price string: "$1,000" -> 1000.0
            price = float(price_str.replace("$", "").replace(",", ""))
            play_percentage = stats["play_percentage"] / 100.0
            prices.append(price)
            play_percentages.append(play_percentage)

        # Sort by price
        sorted_indices = np.argsort(prices)
        prices = np.array(prices)[sorted_indices]
        play_percentages = np.array(play_percentages)[sorted_indices]

        return prices, play_percentages, model_name

    @staticmethod
    def utility_function(wealth: float, gamma: float) -> float:
        """CRRA utility function: U(w) = w^(1-γ)/(1-γ)"""
        if abs(gamma - 1.0) < 1e-6:
            return np.log(max(wealth, 1e-6))
        else:
            if wealth <= 0:
                return -1e10
            return (wealth ** (1 - gamma)) / (1 - gamma)

    def expected_utility_st_petersburg(
        self,
        entry_price: float,
        gamma: float,
        initial_wealth: float = 1000.0,
        max_rounds: int = 30,
    ) -> float:
        """Calculate expected utility for St. Petersburg game"""
        eu = 0.0

        for n in range(1, max_rounds + 1):
            probability = 0.5**n
            payoff = 2 ** (n - 1)
            final_wealth = initial_wealth - entry_price + payoff

            if final_wealth > 0:
                utility = self.utility_function(final_wealth, gamma)
                eu += probability * utility
            else:
                # Loss case - severe negative utility
                eu += probability * (-1e6)

        return eu

    def estimate_crra_parameter(
        self, prices: np.array, play_rates: np.array
    ) -> Tuple[float, float]:
        """Estimate CRRA parameter using maximum likelihood estimation"""

        def negative_log_likelihood(gamma):
            """Negative log-likelihood for binary choice model"""
            if gamma <= 0.01 or gamma > 10:  # Reasonable bounds
                return 1e10

            log_likelihood = 0.0
            initial_wealth = 1000.0

            for price, play_rate in zip(prices, play_rates):
                # Avoid extreme values
                play_rate = max(0.001, min(0.999, play_rate))

                # Expected utility of playing
                eu_play = self.expected_utility_st_petersburg(
                    price, gamma, initial_wealth
                )

                # Expected utility of not playing (keep initial wealth)
                eu_no_play = self.utility_function(initial_wealth, gamma)

                # Utility difference (scaled for numerical stability)
                utility_diff = (
                    (eu_play - eu_no_play) / abs(eu_no_play) if eu_no_play != 0 else 0
                )
                utility_diff = max(-50, min(50, utility_diff))  # Prevent overflow

                # Logistic choice probability
                prob_play = 1.0 / (1.0 + np.exp(-utility_diff))

                # Add to log likelihood
                log_likelihood += play_rate * np.log(prob_play + 1e-10) + (
                    1 - play_rate
                ) * np.log(1 - prob_play + 1e-10)

            return -log_likelihood

        # Try multiple starting points to avoid local minima
        best_gamma = None
        best_likelihood = float("inf")

        for start_gamma in [0.5, 1.0, 1.5, 2.0, 3.0]:
            try:
                result = minimize(
                    negative_log_likelihood,
                    x0=start_gamma,
                    bounds=[(0.01, 9.99)],
                    method="L-BFGS-B",
                )

                if result.success and result.fun < best_likelihood:
                    best_likelihood = result.fun
                    best_gamma = result.x[0]
            except:
                continue

        return best_gamma, best_likelihood if best_gamma else (None, None)

    def estimate_discount_rate(self, prices: np.array, play_rates: np.array) -> float:
        """Estimate implicit discount rate from declining play rates"""
        # Filter out extreme values
        valid_indices = (play_rates > 0.01) & (play_rates < 0.99) & (prices > 0)
        if np.sum(valid_indices) < 3:
            return None

        valid_prices = prices[valid_indices]
        valid_rates = play_rates[valid_indices]

        try:
            # Log-linear relationship between price and willingness to play
            log_prices = np.log(valid_prices)
            logit_rates = np.log(valid_rates / (1 - valid_rates))  # Logit transform

            # Linear regression
            coefficients = np.polyfit(log_prices, logit_rates, 1)
            slope = coefficients[0]

            # Convert slope to implied discount rate
            # Negative slope indicates higher discounting (less willing to pay more)
            discount_rate = max(0, -slope * 0.05)  # Scale factor

            return min(discount_rate, 1.0)  # Cap at 100%
        except:
            return None

    def calculate_certainty_equivalent(
        self, prices: np.array, play_rates: np.array
    ) -> float:
        """Calculate certainty equivalent - maximum price willing to pay"""
        # Find the price where play rate drops below 50%
        for i, rate in enumerate(play_rates):
            if rate < 0.5:
                if i == 0:
                    return prices[0]
                # Interpolate between the two points
                p1, r1 = prices[i - 1], play_rates[i - 1]
                p2, r2 = prices[i], play_rates[i]
                if r1 != r2:
                    ce = p1 + (0.5 - r1) * (p2 - p1) / (r2 - r1)
                    return ce
                else:
                    return p1

        # If always above 50%, return highest price tested
        return prices[-1] if len(prices) > 0 else 0

    def analyze_all_models(self) -> Dict:
        """Main analysis function"""
        results = {}

        print("=== ANALYZING ST. PETERSBURG PARADOX RESULTS ===\n")

        for model_data in self.raw_data:
            prices, play_rates, model_name = self.extract_play_data(model_data)

            print(f"Analyzing {model_name}...")

            # Estimate CRRA parameter
            crra, likelihood = self.estimate_crra_parameter(prices, play_rates)

            # Estimate discount rate
            discount_rate = self.estimate_discount_rate(prices, play_rates)

            # Calculate certainty equivalent
            certainty_equivalent = self.calculate_certainty_equivalent(
                prices, play_rates
            )

            # Calculate summary statistics
            overall_play_rate = (
                model_data["summary_statistics"]["play_percentage"] / 100.0
            )

            results[model_name] = {
                "crra_coefficient": crra,
                "log_likelihood": -likelihood if likelihood else None,
                "discount_rate": discount_rate,
                "certainty_equivalent": certainty_equivalent,
                "overall_play_rate": overall_play_rate,
                "risk_profile": self.classify_risk_profile(crra) if crra else "Unknown",
                "prices": prices.tolist(),
                "play_rates": play_rates.tolist(),
            }

            # Print individual results
            gamma_str = f"{crra:.3f}" if crra else "N/A"
            discount_str = f"{discount_rate:.3f}" if discount_rate else "N/A"
            ce_str = f"${certainty_equivalent:.2f}" if certainty_equivalent else "N/A"

            print(f"  CRRA γ: {gamma_str}")
            print(f"  Discount Rate: {discount_str}")
            print(f"  Certainty Equivalent: {ce_str}")
            print(f"  Risk Profile: {results[model_name]['risk_profile']}")
            print(f"  Overall Play Rate: {overall_play_rate:.1%}\n")

        self.results = results
        return results

    @staticmethod
    def classify_risk_profile(gamma: float) -> str:
        """Classify risk profile based on CRRA coefficient"""
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

    def create_summary_plot(self, save_path: str = None):
        """Create visualization of results"""
        if not self.results:
            print("No results to plot. Run analyze_all_models() first.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        models = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        # Plot 1: CRRA coefficients
        crra_values = [self.results[m]["crra_coefficient"] for m in models]
        crra_values = [v if v is not None else 0 for v in crra_values]

        bars1 = ax1.bar(models, crra_values, color=colors)
        ax1.set_title("CRRA Risk Aversion Coefficients (γ)")
        ax1.set_ylabel("CRRA Coefficient")
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Discount rates
        discount_values = [self.results[m]["discount_rate"] for m in models]
        discount_values = [v if v is not None else 0 for v in discount_values]

        bars2 = ax2.bar(models, discount_values, color=colors)
        ax2.set_title("Estimated Discount Rates")
        ax2.set_ylabel("Discount Rate")
        ax2.tick_params(axis="x", rotation=45)

        # Plot 3: Certainty equivalents
        ce_values = [self.results[m]["certainty_equivalent"] for m in models]

        bars3 = ax3.bar(models, ce_values, color=colors)
        ax3.set_title("Certainty Equivalents")
        ax3.set_ylabel("Maximum Willing to Pay ($)")
        ax3.tick_params(axis="x", rotation=45)
        ax3.set_yscale("log")

        # Plot 4: Play rate curves
        for i, model in enumerate(models):
            prices = self.results[model]["prices"]
            rates = self.results[model]["play_rates"]
            ax4.plot(prices, rates, "o-", color=colors[i], label=model, linewidth=2)

        ax4.set_title("Play Rate vs Entry Price")
        ax4.set_xlabel("Entry Price ($)")
        ax4.set_ylabel("Play Rate")
        ax4.set_xscale("log")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(
        self, filename: str = "st_petersburg_analysis.csv"
    ) -> pd.DataFrame:
        """Export results to CSV"""
        if not self.results:
            print("No results to export. Run analyze_all_models() first.")
            return None

        # Prepare data for export
        export_data = {}
        for model, data in self.results.items():
            export_data[model] = {
                "CRRA_Coefficient": data["crra_coefficient"],
                "Discount_Rate": data["discount_rate"],
                "Certainty_Equivalent": data["certainty_equivalent"],
                "Overall_Play_Rate": data["overall_play_rate"],
                "Risk_Profile": data["risk_profile"],
                "Log_Likelihood": data["log_likelihood"],
            }

        df = pd.DataFrame.from_dict(export_data, orient="index")
        df.to_csv(filename, index_label="Model")

        print(f"Results exported to {filename}")
        print("\nSummary Table:")
        print(df.round(3))

        return df


def main():
    """Main execution function"""

    # Initialize analyzer
    analyzer = StPetersburgAnalyzer()

    # Load JSON files from results directory
    results_dir = (
        "/Users/arianakbari/Desktop/research/research/st.petersburg-paradox/results"
    )
    analyzer.load_json_results(results_dir)

    # Analyze all models
    results = analyzer.analyze_all_models()

    # Create visualization
    plot_path = "/Users/arianakbari/Desktop/research/research/st.petersburg-paradox/analyze/st_petersburg_analysis.png"
    analyzer.create_summary_plot(plot_path)

    # Export results to CSV
    csv_path = "/Users/arianakbari/Desktop/research/research/st.petersburg-paradox/analyze/st_petersburg_analysis.csv"
    df = analyzer.export_results(csv_path)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS SUMMARY")
    print("=" * 70)

    for model, data in results.items():
        crra = f"{data['crra_coefficient']:.3f}" if data["crra_coefficient"] else "N/A"
        discount = f"{data['discount_rate']:.3f}" if data["discount_rate"] else "N/A"
        ce = (
            f"${data['certainty_equivalent']:.2f}"
            if data["certainty_equivalent"]
            else "N/A"
        )

        print(f"\n{model}:")
        print(f"  • CRRA γ: {crra} ({data['risk_profile']})")
        print(f"  • Discount Rate: {discount}")
        print(f"  • Certainty Equivalent: {ce}")
        print(f"  • Overall Play Rate: {data['overall_play_rate']:.1%}")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
