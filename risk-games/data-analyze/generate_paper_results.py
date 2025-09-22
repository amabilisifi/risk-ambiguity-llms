"""
Academic Paper Results Generator for AI Risk Aversion Study
Generates publication-ready tables and visualizations from CARA and CRRA analyses
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up professional plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "font.family": "serif",
    }
)


class AcademicResultsGenerator:
    def __init__(
        self,
        analysis_dir="analysis",
        output_dir="paper_outputs",
        normalize=False,
        norm_method="zscore",
    ):
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Normalization settings for additional comparative plots
        self.normalize = normalize
        self.norm_method = norm_method  # 'zscore' or 'minmax'

        # Model mapping for publication - focused on O3 Mini Opportunity Hunter
        self.model_names = {
            "o3-mini-opportunity-hunter": "O3 Mini (Opportunity Hunter)",
        }

    def load_results(self):
        """Load O3 Mini CARA and CRRA results from JSON files"""
        self.cara_data = {}
        self.crra_data = {}

        print(f"🔍 Loading O3 Mini results from: {self.analysis_dir}")

        # Look for specific O3 Mini Opportunity Hunter result files
        # Find the most recent opportunity hunter result files
        cara_pattern = "cara_results_o3-mini_opportunity_hunter_*.json"
        crra_pattern = "crra_results_o3-mini_opportunity_hunter_*.json"

        cara_files = list(self.analysis_dir.glob(cara_pattern))
        crra_files = list(self.analysis_dir.glob(crra_pattern))

        cara_file = (
            max(cara_files, key=lambda f: f.stat().st_mtime) if cara_files else None
        )
        crra_file = (
            max(crra_files, key=lambda f: f.stat().st_mtime) if crra_files else None
        )

        # Load CARA results
        if cara_file and cara_file.exists():
            try:
                with open(cara_file, "r") as f:
                    data = json.load(f)
                    model = data.get("model", "o3-mini-opportunity-hunter")
                    self.cara_data[model] = data
                print(
                    f"✅ Loaded CARA results for O3 Mini (Opportunity Hunter): {cara_file.name}"
                )
            except Exception as e:
                print(f"❌ Error loading CARA file: {e}")
        else:
            print(f"❌ CARA results file not found")

        # Load CRRA results
        if crra_file and crra_file.exists():
            try:
                with open(crra_file, "r") as f:
                    data = json.load(f)
                    model = data.get("model", "o3-mini-opportunity-hunter")
                    self.crra_data[model] = data
                print(
                    f"✅ Loaded CRRA results for O3 Mini (Opportunity Hunter): {crra_file.name}"
                )
            except Exception as e:
                print(f"❌ Error loading CRRA file: {e}")
        else:
            print(f"❌ CRRA results file not found")

        print(
            f"✅ Loaded results for O3 Mini: {len(self.cara_data)} CARA, {len(self.crra_data)} CRRA"
        )

    def generate_cara_table(self):
        """Generate professional CARA results table"""
        cara_summary = []

        for model_key, data in self.cara_data.items():
            model_name = self.model_names.get(model_key, model_key)

            cara_summary.append(
                {
                    "Model": model_name,
                    "Risk Aversion (α)": f"{data['parameters']['risk_aversion_alpha']:.4f}",
                    "Choice Sensitivity (β)": f"{data['parameters']['choice_sensitivity_beta']:.2f}",
                    "Risk Classification": data["interpretation"]["risk_type"],
                    "Risky Choices": f"{data['data_summary']['overall_risky_rate']*100:.1f}%",
                    "Log-Likelihood": f"{data['fit_quality']['negative_log_likelihood']:.2f}",
                }
            )

        cara_df = pd.DataFrame(cara_summary)

        # Save as CSV and styled table
        cara_df.to_csv(self.output_dir / "cara_results_table.csv", index=False)

        # Create styled table image (only if there's data)
        if len(cara_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis("tight")
            ax.axis("off")

            table = ax.table(
                cellText=cara_df.values,
                colLabels=cara_df.columns,
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2)

            # Style the table
            for i in range(len(cara_df.columns)):
                table[(0, i)].set_facecolor("#4472C4")
                table[(0, i)].set_text_props(weight="bold", color="white")

            plt.title(
                "CARA Risk Aversion Analysis Results",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            plt.savefig(
                self.output_dir / "cara_table.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        else:
            print("⚠️  No CARA data found - skipping table generation")

        return cara_df

    def generate_crra_table(self):
        """Generate professional CRRA results table"""
        crra_summary = []

        for model_key, data in self.crra_data.items():
            model_name = self.model_names.get(model_key, model_key)

            # Calculate risky choice percentage
            total_risky = sum(t["risky_choices"] for t in data["raw_trials"])
            total_choices = sum(t["total_choices"] for t in data["raw_trials"])
            risky_pct = (total_risky / total_choices) * 100

            crra_summary.append(
                {
                    "Model": model_name,
                    "Risk Aversion (r)": f"{data['parameters']['risk_aversion_r']:.4f}",
                    "Choice Sensitivity (β)": f"{data['parameters']['choice_sensitivity_beta']:.2f}",
                    "Risk Classification": data["interpretation"]["risk_type"],
                    "Risky Choices": f"{risky_pct:.1f}%",
                    "Log-Likelihood": f"{data['fit_quality']['negative_log_likelihood']:.2f}",
                }
            )

        crra_df = pd.DataFrame(crra_summary)

        # Save as CSV and styled table
        crra_df.to_csv(self.output_dir / "crra_results_table.csv", index=False)

        # Create styled table image (only if there's data)
        if len(crra_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis("tight")
            ax.axis("off")

            table = ax.table(
                cellText=crra_df.values,
                colLabels=crra_df.columns,
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2)

            # Style the table
            for i in range(len(crra_df.columns)):
                table[(0, i)].set_facecolor("#C55A5A")
                table[(0, i)].set_text_props(weight="bold", color="white")

            plt.title(
                "CRRA Risk Aversion Analysis Results",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            plt.savefig(
                self.output_dir / "crra_table.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        else:
            print("⚠️  No CRRA data found - skipping table generation")

        return crra_df

    def generate_risk_aversion_comparison(self):
        """Display O3 Mini risk aversion parameters"""
        # Skip if no data available
        if not self.cara_data or not self.crra_data:
            print("⚠️  Insufficient data for risk aversion comparison - skipping")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # O3 Mini CARA Risk Aversion
        model_key = list(self.cara_data.keys())[0]  # Should be "o3-mini"
        cara_alpha = self.cara_data[model_key]["parameters"]["risk_aversion_alpha"]
        cara_beta = self.cara_data[model_key]["parameters"]["choice_sensitivity_beta"]
        model_label = self.model_names.get(model_key, model_key)

        # Create single bar for CARA
        bars1 = ax1.bar(
            [model_label],
            [cara_alpha],
            color="#4472C4",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
            width=0.6,
        )
        ax1.set_title("O3 Mini CARA Risk Aversion (α)", fontweight="bold")
        ax1.set_ylabel("Alpha Coefficient")
        ax1.set_xlabel("Model")
        ax1.set_ylim(0, max(cara_alpha * 1.2, 0.1))  # Set reasonable y-axis limit

        # Add value label
        height = cara_alpha
        ax1.text(
            0,
            height + height * 0.05,
            f"{cara_alpha:.6f}\n(β = {cara_beta:.2f})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

        # O3 Mini CRRA Risk Aversion
        crra_r = self.crra_data[model_key]["parameters"]["risk_aversion_r"]
        crra_beta = self.crra_data[model_key]["parameters"]["choice_sensitivity_beta"]

        # Create single bar for CRRA
        bars2 = ax2.bar(
            [model_label],
            [crra_r],
            color="#C55A5A",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
            width=0.6,
        )
        ax2.set_title("O3 Mini CRRA Risk Aversion (r)", fontweight="bold")
        ax2.set_ylabel("r Coefficient")
        ax2.set_xlabel("Model")
        ax2.set_ylim(0, max(crra_r * 1.2, 0.5))  # Set reasonable y-axis limit

        # Add value label
        height = crra_r
        ax2.text(
            0,
            height + height * 0.05,
            f"{crra_r:.4f}\n(β = {crra_beta:.2f})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "risk_aversion_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        # Add interpretation text
        fig.suptitle(
            "O3 Mini Risk Aversion Analysis", fontsize=16, fontweight="bold", y=0.98
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "o3_mini_risk_aversion.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_choice_sensitivity_comparison(self):
        """Display O3 Mini choice sensitivity parameters"""
        # Skip if no data available
        if not self.cara_data or not self.crra_data:
            print("⚠️  Insufficient data for choice sensitivity comparison - skipping")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        models = list(self.cara_data.keys())
        model_labels = [self.model_names.get(m, m) for m in models]

        # CARA Choice Sensitivity
        cara_betas = [
            self.cara_data[m]["parameters"]["choice_sensitivity_beta"] for m in models
        ]

        bars1 = ax1.bar(
            model_labels,
            cara_betas,
            color=["#9DC3E6", "#F4B183", "#A9D18E", "#FFE699"],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        ax1.set_title("CARA Choice Sensitivity (β)", fontweight="bold")
        ax1.set_ylabel("Beta Coefficient")
        ax1.set_xlabel("AI Model")

        # Add value labels on bars
        for bar, val in zip(bars1, cara_betas):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # CRRA Choice Sensitivity
        crra_betas = [
            self.crra_data[m]["parameters"]["choice_sensitivity_beta"] for m in models
        ]

        bars2 = ax2.bar(
            model_labels,
            crra_betas,
            color=["#9DC3E6", "#F4B183", "#A9D18E", "#FFE699"],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        ax2.set_title("CRRA Choice Sensitivity (β)", fontweight="bold")
        ax2.set_ylabel("Beta Coefficient")
        ax2.set_xlabel("AI Model")

        # Add value labels on bars
        for bar, val in zip(bars2, crra_betas):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "o3_mini_choice_sensitivity.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        if self.normalize:
            self._generate_normalized_parameter_plot(
                param_keys=[
                    ("CARA", "choice_sensitivity_beta"),
                    ("CRRA", "choice_sensitivity_beta"),
                ],
                title="Normalized Choice Sensitivity (β) Across Models",
                filename="choice_sensitivity_comparison_normalized.png",
                colors=["#9DC3E6", "#E7745A"],
            )

    def _normalize_series(self, values):
        arr = np.array(values, dtype=float)
        if self.norm_method == "zscore":
            mu = np.nanmean(arr)
            sd = np.nanstd(arr)
            if sd == 0:
                sd = 1.0
            return (arr - mu) / sd, f"z-score (μ={mu:.3g}, σ={sd:.3g})"
        elif self.norm_method == "minmax":
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            rng = mx - mn
            if rng == 0:
                rng = 1.0
            return (arr - mn) / rng, f"min-max (min={mn:.3g}, max={mx:.3g})"
        else:
            raise ValueError(f"Unsupported norm method: {self.norm_method}")

    def _generate_normalized_parameter_plot(self, param_keys, title, filename, colors):
        """Create grouped bar plot with normalized metrics for each model.

        param_keys: list of tuples (family, key) where family is 'CARA' or 'CRRA'.
        """
        models = list(self.cara_data.keys())
        if not models:
            return
        model_labels = [self.model_names.get(m, m) for m in models]
        # Collect values per parameter set
        all_normed = []
        norm_labels = []
        for fam, key in param_keys:
            vals = []
            for m in models:
                if fam == "CARA":
                    vals.append(self.cara_data[m]["parameters"][key])
                else:
                    vals.append(self.crra_data[m]["parameters"][key])
            normed, norm_desc = self._normalize_series(vals)
            all_normed.append(normed)
            norm_labels.append(f"{fam} {key.split('_')[-1]}\n{norm_desc}")

        x = np.arange(len(models))
        width = 0.35 if len(all_normed) == 2 else 0.8 / len(all_normed)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, norm_vals in enumerate(all_normed):
            offset = (i - (len(all_normed) - 1) / 2) * width * 1.1
            ax.bar(
                x + offset,
                norm_vals,
                width,
                color=colors[i % len(colors)],
                alpha=0.8,
                label=param_keys[i][0],
            )
            for xi, val in zip(x + offset, norm_vals):
                ax.text(
                    xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=9
                )
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=0)
        ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_ylabel(f"Normalized Value ({self.norm_method})")
        ax.set_title(title, fontweight="bold")
        ax.legend(title="Parameter Family")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_behavioral_profile_heatmap(self):
        """Generate O3 Mini behavioral profile heatmap"""
        # Skip if no data available
        if not self.cara_data or not self.crra_data:
            print("⚠️  Insufficient data for behavioral profile heatmap - skipping")
            return

        models = list(self.cara_data.keys())
        model_labels = [self.model_names.get(m, m) for m in models]

        # Collect all metrics
        metrics_data = []
        for model in models:
            cara = self.cara_data[model]
            crra = self.crra_data[model]

            # Calculate risky choice percentage for CRRA
            total_risky = sum(t["risky_choices"] for t in crra["raw_trials"])
            total_choices = sum(t["total_choices"] for t in crra["raw_trials"])
            risky_pct = (total_risky / total_choices) * 100

            metrics_data.append(
                [
                    cara["parameters"]["risk_aversion_alpha"],
                    cara["parameters"]["choice_sensitivity_beta"],
                    crra["parameters"]["risk_aversion_r"],
                    crra["parameters"]["choice_sensitivity_beta"],
                    risky_pct,
                    cara["data_summary"]["overall_risky_rate"] * 100,
                ]
            )

        metrics_df = pd.DataFrame(
            metrics_data,
            index=model_labels,
            columns=[
                "CARA α",
                "CARA β",
                "CRRA r",
                "CRRA β",
                "Risky Choices (%)",
                "CARA Risky (%)",
            ],
        )

        # Normalize for heatmap (z-score)
        metrics_normalized = (metrics_df - metrics_df.mean()) / metrics_df.std()

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            metrics_normalized.T,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".2f",
            cbar_kws={"label": "Standardized Score"},
        )
        plt.title(
            "O3 Mini Risk Preference Behavioral Profile",
            fontweight="bold",
            fontsize=16,
        )
        plt.xlabel("O3 Mini Model")
        plt.ylabel("Risk Preference Metrics")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "o3_mini_behavioral_profile_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_risk_choice_scatter(self):
        """Generate O3 Mini risk aversion vs choice behavior scatter plot"""
        # Skip if no data available
        if not self.cara_data or not self.crra_data:
            print("⚠️  Insufficient data for risk choice scatter plot - skipping")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        models = list(self.cara_data.keys())
        model_labels = [self.model_names.get(m, m) for m in models]
        colors = ["#4472C4", "#E7745A", "#70AD47", "#FFC000"]

        # CARA: Risk Aversion vs Risky Choices
        cara_alphas = [
            self.cara_data[m]["parameters"]["risk_aversion_alpha"] for m in models
        ]
        cara_risky_pcts = [
            self.cara_data[m]["data_summary"]["overall_risky_rate"] * 100
            for m in models
        ]

        for i, (alpha, risky_pct, label) in enumerate(
            zip(cara_alphas, cara_risky_pcts, model_labels)
        ):
            ax1.scatter(
                alpha,
                risky_pct,
                s=200,
                c=colors[i],
                alpha=0.7,
                edgecolor="black",
                linewidth=2,
            )
            ax1.annotate(
                label,
                (alpha, risky_pct),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold",
                fontsize=10,
            )

        ax1.set_xlabel("CARA Risk Aversion (α)")
        ax1.set_ylabel("Risky Choices (%)")
        ax1.set_title("CARA: Risk Aversion vs Risk-Taking", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # CRRA: Risk Aversion vs Risky Choices
        crra_rs = [self.crra_data[m]["parameters"]["risk_aversion_r"] for m in models]
        crra_risky_pcts = []
        for m in models:
            total_risky = sum(
                t["risky_choices"] for t in self.crra_data[m]["raw_trials"]
            )
            total_choices = sum(
                t["total_choices"] for t in self.crra_data[m]["raw_trials"]
            )
            crra_risky_pcts.append((total_risky / total_choices) * 100)

        for i, (r, risky_pct, label) in enumerate(
            zip(crra_rs, crra_risky_pcts, model_labels)
        ):
            ax2.scatter(
                r,
                risky_pct,
                s=200,
                c=colors[i],
                alpha=0.7,
                edgecolor="black",
                linewidth=2,
            )
            ax2.annotate(
                label,
                (r, risky_pct),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold",
                fontsize=10,
            )

        ax2.set_xlabel("CRRA Risk Aversion (r)")
        ax2.set_ylabel("Risky Choices (%)")
        ax2.set_title("CRRA: Risk Aversion vs Risk-Taking", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "risk_aversion_vs_behavior.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_summary_report(self):
        """Generate text summary for paper"""
        report = []
        report.append("# O3 Mini (Opportunity Hunter) Risk Aversion Analysis Summary")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Key Findings for O3 Mini (Opportunity Hunter):")

        # Get O3 Mini data
        model_key = list(self.cara_data.keys())[0]
        cara_data = self.cara_data[model_key]
        crra_data = self.crra_data[model_key]

        report.append("\n### O3 Mini (Opportunity Hunter) CARA Analysis:")
        alpha = cara_data["parameters"]["risk_aversion_alpha"]
        beta_cara = cara_data["parameters"]["choice_sensitivity_beta"]
        risk_type_cara = cara_data["interpretation"]["risk_type"]
        risky_pct_cara = cara_data["data_summary"]["overall_risky_rate"] * 100
        nll_cara = cara_data["fit_quality"]["negative_log_likelihood"]

        report.append(f"• Risk Aversion (α): {alpha:.6f}")
        report.append(f"• Choice Sensitivity (β): {beta_cara:.2f}")
        report.append(f"• Risk Classification: {risk_type_cara}")
        report.append(f"• Risky Choices: {risky_pct_cara:.1f}%")
        report.append(f"• Model Fit (NLL): {nll_cara:.2f}")

        report.append("\n### O3 Mini (Opportunity Hunter) CRRA Analysis:")
        r = crra_data["parameters"]["risk_aversion_r"]
        beta_crra = crra_data["parameters"]["choice_sensitivity_beta"]
        risk_type_crra = crra_data["interpretation"]["risk_type"]
        total_risky = sum(t["risky_choices"] for t in crra_data["raw_trials"])
        total_choices = sum(t["total_choices"] for t in crra_data["raw_trials"])
        risky_pct_crra = (total_risky / total_choices) * 100
        nll_crra = crra_data["fit_quality"]["negative_log_likelihood"]

        report.append(f"• Risk Aversion (r): {r:.4f}")
        report.append(f"• Choice Sensitivity (β): {beta_crra:.2f}")
        report.append(f"• Risk Classification: {risk_type_crra}")
        report.append(f"• Risky Choices: {risky_pct_crra:.1f}%")
        report.append(f"• Model Fit (NLL): {nll_crra:.2f}")

        report.append("\n### Comparative Analysis:")
        report.append(f"• CARA vs CRRA Risk Aversion: α = {alpha:.6f} vs r = {r:.4f}")
        report.append(
            f"• Risk Classification Consistency: {'Consistent' if risk_type_cara == risk_type_crra else 'Different'}"
        )
        report.append(
            f"• Choice Pattern: {risky_pct_cara:.1f}% risky choices (both models)"
        )
        report.append(
            f"• Model Fit: CARA NLL = {nll_cara:.2f}, CRRA NLL = {nll_crra:.2f}"
        )

        report_text = "\n".join(report)

        with open(self.output_dir / "analysis_summary.txt", "w") as f:
            f.write(report_text)

        return report_text

    def run_all_analyses(self):
        """Run complete analysis pipeline"""
        print("🎯 GENERATING O3 MINI (OPPORTUNITY HUNTER) ACADEMIC PAPER RESULTS")
        print("=" * 50)

        # Load data
        self.load_results()

        # Generate tables
        print("📊 Generating results tables...")
        cara_df = self.generate_cara_table()
        crra_df = self.generate_crra_table()

        # Generate visualizations
        print("📈 Creating visualizations...")
        self.generate_risk_aversion_comparison()
        self.generate_choice_sensitivity_comparison()
        self.generate_behavioral_profile_heatmap()
        self.generate_risk_choice_scatter()

        # Generate summary
        print("📝 Creating summary report...")
        summary = self.generate_summary_report()

        print(f"\n✅ ALL OUTPUTS GENERATED!")
        print(f"📁 Results saved in: {self.output_dir}")
        print("\n📋 Files created:")

        output_files = [
            "cara_table.png - O3 Mini CARA results table",
            "crra_table.png - O3 Mini CRRA results table",
            "cara_results_table.csv - O3 Mini CARA data (CSV)",
            "crra_results_table.csv - O3 Mini CRRA data (CSV)",
            "o3_mini_risk_aversion.png - O3 Mini Risk aversion analysis",
            "o3_mini_choice_sensitivity.png - O3 Mini Choice sensitivity analysis",
            "o3_mini_behavioral_profile_heatmap.png - O3 Mini behavioral profiles",
            "risk_aversion_vs_behavior.png - O3 Mini Risk aversion vs behavior scatter",
            "analysis_summary.txt - O3 Mini text summary",
        ]

        if self.normalize:
            # Add normalized output file names
            output_files.extend(
                [
                    "risk_aversion_comparison_normalized.png - Normalized risk aversion (grouped)",
                    "choice_sensitivity_comparison_normalized.png - Normalized choice sensitivity",
                ]
            )

        for file in output_files:
            print(f"   • {file}")

        print(
            f"\n🎓 O3 Mini (Opportunity Hunter) results ready for academic paper inclusion!"
        )

        return {
            "cara_df": cara_df,
            "crra_df": crra_df,
            "summary": summary,
            "output_dir": self.output_dir,
        }


# Run the analysis
if __name__ == "__main__":
    import sys

    # Parse command line arguments
    analysis_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/arianakbari/Desktop/research/research/ambiguity-aversion/data-analyze/analysis"
    )
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "paper_outputs"

    generator = AcademicResultsGenerator(
        analysis_dir=analysis_dir, output_dir=output_dir
    )
    results = generator.run_all_analyses()
