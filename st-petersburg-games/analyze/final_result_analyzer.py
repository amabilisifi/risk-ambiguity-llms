import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob
import numpy as np
from typing import Dict, List

# Configuration
# Resolve base directory relative to this file's location to avoid path issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "final-result")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analyze/analysis_results")

# Create output directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_all_model_results() -> Dict:
    """Load results for all models from the analysis_results directory"""
    results = {}
    pattern = os.path.join(ANALYSIS_DIR, "behavioral_analysis_*.json")
    for filepath in glob.glob(pattern):
        try:
            model_name = os.path.basename(filepath).split('_')[3]  # Extract model name
            with open(filepath, 'r') as f:
                results[model_name] = json.load(f)
            print(f"Successfully loaded data for model: {model_name}")
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
    return results

def analyze_justification(text: str) -> Dict:
    """Analyze a single justification text"""
    return {
        'decision': text.split('.')[0].strip().upper(),
        'reasoning_length': len(text),
        'mentions_infinite_ev': 'infinite' in text.lower(),
        'mentions_utility': 'utility' in text.lower(),
        'mentions_risk': 'risk' in text.lower(),
        'mentions_probability': 'probability' in text.lower(),
        'mentions_mathematical': 'mathematical' in text.lower(),
        'mentions_constraints': any(word in text.lower() for word in ['constraint', 'limit', 'bound']),
        'full_text': text
    }

def create_comparative_analysis(all_results: Dict) -> pd.DataFrame:
    """Create detailed comparative analysis across models"""
    model_analyses = {}
    
    for model, data in all_results.items():
        # Two schema support:
        # 1) Flat schema with 'detailed_results' and 'final_justification'
        # 2) Rich schema under 'questions_analyzed'
        if 'questions_analyzed' in data:
            qa = data['questions_analyzed']
            # Infinity understanding section
            inf_sec = qa.get('infinity_understanding_vs_behavior', {})
            inf_scores = inf_sec.get('score_distribution', {})
            total_inf = max(1, inf_sec.get('valid_analyses', 0) or sum(inf_scores.values()) or 1)
            understands_inf_pct = (inf_scores.get('score_2', 0) / total_inf) * 100
            # Collect texts for keyword analysis
            texts = []
            for sec_key in ['infinity_understanding_vs_behavior', 'breakpoint_decision_drivers']:
                sec = qa.get(sec_key, {})
                for it in sec.get('detailed_results', []):
                    texts.append(it.get('explanation', '') or '')
            anal = [analyze_justification(t) for t in texts]
            df = pd.DataFrame(anal) if anal else pd.DataFrame(columns=list(analyze_justification('').keys()))
            # Breakpoint rationality (average score in breakpoint section if present)
            bp_sec = qa.get('breakpoint_decision_drivers', {})
            bp_scores = bp_sec.get('score_distribution', {})
            total_bp = max(1, bp_sec.get('valid_analyses', 0) or sum(bp_scores.values()) or 1)
            bp_avg_score = bp_sec.get('average_score', None)
            model_analyses[model] = {
                'total_decisions': len(df),
                'pass_rate': np.nan,  # not available in this schema
                'avg_reasoning_length': float(df['reasoning_length'].mean()) if not df.empty else np.nan,
                'infinite_ev_mentions': float(df['mentions_infinite_ev'].mean() * 100) if not df.empty else 0.0,
                'utility_mentions': float(df['mentions_utility'].mean() * 100) if not df.empty else 0.0,
                'risk_mentions': float(df['mentions_risk'].mean() * 100) if not df.empty else 0.0,
                'probability_mentions': float(df['mentions_probability'].mean() * 100) if not df.empty else 0.0,
                'mathematical_mentions': float(df['mentions_mathematical'].mean() * 100) if not df.empty else 0.0,
                'constraints_mentions': float(df['mentions_constraints'].mean() * 100) if not df.empty else 0.0,
                'understands_infinite_ev_pct': understands_inf_pct,
                'breakpoint_rationality_avg': bp_avg_score if bp_avg_score is not None else (bp_scores.get('score_2', 0)*2 + bp_scores.get('score_1', 0)*1) / total_bp,
            }
        else:
            justifications = [res.get('final_justification', '') 
                             for res in data.get('detailed_results', [])]
            analyses = [analyze_justification(j) for j in justifications]
            df = pd.DataFrame(analyses)
            model_analyses[model] = {
                'total_decisions': len(df),
                'pass_rate': (df['decision'].str.contains('PASS')).mean() * 100 if not df.empty else np.nan,
                'avg_reasoning_length': df['reasoning_length'].mean() if not df.empty else np.nan,
                'infinite_ev_mentions': df['mentions_infinite_ev'].mean() * 100 if not df.empty else 0.0,
                'utility_mentions': df['mentions_utility'].mean() * 100 if not df.empty else 0.0,
                'risk_mentions': df['mentions_risk'].mean() * 100 if not df.empty else 0.0,
                'probability_mentions': df['mentions_probability'].mean() * 100 if not df.empty else 0.0,
                'mathematical_mentions': df['mentions_mathematical'].mean() * 100 if not df.empty else 0.0,
                'constraints_mentions': df['mentions_constraints'].mean() * 100 if not df.empty else 0.0,
                'understands_infinite_ev_pct': np.nan,
                'breakpoint_rationality_avg': np.nan,
            }
    
    return pd.DataFrame(model_analyses).T

def create_visualizations(comparison_df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations"""
    # Set style via seaborn
    sns.set_theme()
    
    # 1. Decision Patterns
    plt.figure(figsize=(12, 6))
    comparison_df['pass_rate'].plot(kind='bar', color='skyblue')
    plt.title('Pass Rate by Model', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pass_rates_comparison.png'))
    plt.close()
    
    # 2. Concept Analysis Heatmap
    concept_columns = ['infinite_ev_mentions', 'utility_mentions', 'risk_mentions', 
                      'probability_mentions', 'mathematical_mentions', 'constraints_mentions']
    plt.figure(figsize=(12, 8))
    sns.heatmap(comparison_df[concept_columns], annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Concept Mention Rates by Model (%)', fontsize=14)
    plt.xlabel('Concepts', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_analysis_heatmap.png'))
    plt.close()
    
    # 3. Reasoning Length Distribution
    plt.figure(figsize=(12, 6))
    comparison_df['avg_reasoning_length'].plot(kind='bar', color='lightgreen')
    plt.title('Average Reasoning Length by Model', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Characters', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reasoning_length_comparison.png'))
    plt.close()

    # 4. Infinity understanding score (if available)
    if 'understands_infinite_ev_pct' in comparison_df.columns:
        plt.figure(figsize=(12, 6))
        comparison_df['understands_infinite_ev_pct'].plot(kind='bar', color='#8da0cb')
        plt.title('Understands Infinite EV (Score=2) - Rate by Model (%)', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'infinite_ev_understanding_rate.png'))
        plt.close()

    # 5. Risk vs Utility emphasis
    if set(['risk_mentions','utility_mentions']).issubset(comparison_df.columns):
        plt.figure(figsize=(12, 6))
        comparison_df[['risk_mentions','utility_mentions']].plot(kind='bar', figsize=(12,6))
        plt.title('Emphasis: Risk vs Utility Mentions (%)', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Mention Rate (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_vs_utility_mentions.png'))
        plt.close()

def generate_report(comparison_df: pd.DataFrame, all_results: Dict, output_dir: str):
    """Generate comprehensive analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'st_petersburg_analysis_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("St. Petersburg Paradox - Comparative Model Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall Statistics
        f.write("1. Overall Statistics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total models analyzed: {len(comparison_df)}\n")
        f.write(f"Average pass rate across models: {comparison_df['pass_rate'].mean():.2f}%\n")
        f.write(f"Pass rate standard deviation: {comparison_df['pass_rate'].std():.2f}%\n\n")
        
        # Model Comparison
        f.write("2. Model Comparison Metrics\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_df.to_string())
        f.write("\n\n")
        
        # Key Findings
        f.write("3. Key Findings\n")
        f.write("-" * 30 + "\n")
        f.write(f"Most consistent model: {comparison_df['pass_rate'].idxmax()} ")
        f.write(f"(Pass rate: {comparison_df['pass_rate'].max():.2f}%)\n")
        f.write(f"Most detailed responses: {comparison_df['avg_reasoning_length'].idxmax()} ")
        f.write(f"(Avg length: {comparison_df['avg_reasoning_length'].max():.0f} chars)\n\n")
        
        # Concept Analysis
        f.write("4. Concept Analysis\n")
        f.write("-" * 30 + "\n")
        concept_cols = ['infinite_ev_mentions', 'utility_mentions', 'risk_mentions',
                       'probability_mentions', 'mathematical_mentions', 'constraints_mentions']
        for concept in concept_cols:
            f.write(f"{concept.replace('_', ' ').title()}: ")
            f.write(f"Avg: {comparison_df[concept].mean():.2f}%, ")
            f.write(f"Max: {comparison_df[concept].max():.2f}% ({comparison_df[concept].idxmax()})\n")

        # Persian Q&A style summary addressing key questions
        fa_path = os.path.join(output_dir, f'answers_summary_fa_{timestamp}.txt')
    
    # Build Persian answers based on available metrics
    lines_fa: List[str] = []
    lines_fa.append("پاسخ به پرسش‌های تحقیق (پارادوکس سن‌پترزبورگ)\n")
    lines_fa.append("="*50 + "\n\n")
    # Infinity understanding
    if 'understands_infinite_ev_pct' in comparison_df.columns:
        best_model_inf = comparison_df['understands_infinite_ev_pct'].idxmax()
        lines_fa.append(f"- درک بی‌نهایت: میانگین نرخ درک صریح EV بی‌نهایت (امتیاز ۲): {comparison_df['understands_infinite_ev_pct'].mean():.1f}%؛ بهترین مدل: {best_model_inf} با {comparison_df['understands_infinite_ev_pct'].max():.1f}%\n")
    # Risk vs Utility
    if set(['risk_mentions','utility_mentions']).issubset(comparison_df.columns):
        dominant = (comparison_df['risk_mentions'].mean() > comparison_df['utility_mentions'].mean())
        pref = 'ریسک' if dominant else 'مطلوبیت'
        lines_fa.append(f"- تاکید اصلی: میانگین اشاره‌ها نشان می‌دهد تمرکز بیشتر بر «{pref}» بوده است.\n")
    # Probability dispersion
    if 'probability_mentions' in comparison_df.columns:
        lines_fa.append(f"- پراکندگی احتمال: نرخ اشاره به احتمال/توزیع به طور میانگین {comparison_df['probability_mentions'].mean():.1f}% است.\n")
    # Constraints / small outcomes proxy
    if 'constraints_mentions' in comparison_df.columns:
        lines_fa.append(f"- قیود/نتایج کوچک: اشاره به قیود/کوچک بودن نتایج به طور میانگین {comparison_df['constraints_mentions'].mean():.1f}% مشاهده شد.\n")
    # Breakpoint rationality
    if 'breakpoint_rationality_avg' in comparison_df.columns:
        mask = comparison_df['breakpoint_rationality_avg'].notna()
        if mask.any():
            lines_fa.append(f"- عقلانیت نقطه شکست: میانگین امتیاز {comparison_df.loc[mask,'breakpoint_rationality_avg'].mean():.2f} از ۲.\n")

    with open(fa_path, 'w') as ffa:
        ffa.writelines(lines_fa)

def main():
    print("Loading model results...")
    all_results = load_all_model_results()
    
    if not all_results:
        print("Error: No results found to analyze")
        return
    
    print("Creating comparative analysis...")
    comparison_df = create_comparative_analysis(all_results)
    
    print("Generating visualizations...")
    create_visualizations(comparison_df, RESULTS_DIR)
    
    print("Generating comprehensive report...")
    generate_report(comparison_df, all_results, RESULTS_DIR)
    
    print(f"\n✅ Analysis complete! Results saved to: {RESULTS_DIR}")
    print("Generated files:")
    print("- pass_rates_comparison.png")
    print("- concept_analysis_heatmap.png")
    print("- reasoning_length_comparison.png")
    print("- st_petersburg_analysis_[timestamp].txt")

if __name__ == "__main__":
    main()