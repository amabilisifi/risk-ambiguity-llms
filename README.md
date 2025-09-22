# Risk and Ambiguity Analysis for LLMs

This repository provides tools and frameworks for analyzing how Large Language Models (LLMs) handle risk and ambiguity in decision-making tasks. The project includes experimental setups, analysis pipelines, and visualization tools to compare LLM behavior with human decision-making patterns.


## Key Features

- Risk preference measurement using CRRA (Constant Relative Risk Aversion) and CARA (Constant Absolute Risk Aversion) models
- Ambiguity aversion analysis with ε-contamination framework
- Opportunity hunter persona experiments
- Comprehensive visualization and statistical analysis tools
- Automated experiment pipelines for multiple LLM models

## Getting Started

1. Install dependencies:

First, rename the requirements file:

```sh
mv requiremets.txt requirements.txt
```

Then install the dependencies:

```sh
pip install -r requirements.txt
```

2. Set up environment variables:

```sh
cp .env.example .env
# Add your Azure OpenAI API credentials
```

3. Run experiments:

```sh
# Risk preference analysis
python risk-games/risk_game.py

# Ambiguity preference analysis
python ambiguity-games/ambiguity_game.py
```

## Analysis Tools

- `risk-games/data-analyze/fit_crra.py`: CRRA utility model fitting
- `risk-games/data-analyze/fit_cara.py`: CARA utility model fitting
- `risk-games/data-analyze/plot_crra_utility.py`: Utility function visualization
- `risk-games/data-analyze/generate_paper_results.py`: Publication-ready visualizations

## Experimental Results

The project includes comprehensive results for various LLM models:

### Supported Models

- GPT-4 Turbo (gpt-4o)
- GPT-4.1
- GPT-4 Turbo Mini (gpt-4o-mini)
- GPT-5
- O3-Mini

## Project Structure

```
.
├── models.py               # Core model implementations
├── models_enum.py         # Model type enumerations
├── test_model.py          # Model test suite
├── ambiguity-games/       # Ambiguity preference experiments
│   ├── ambiguity_game.py  # Main game implementation
│   ├── run_opportunity_hunter.py
│   └── results/          # Experimental results
│       ├── neutral/      # Results for neutral persona
│       └── persona/      # Results for different personas
├── risk-games/           # Risk preference analysis
│   ├── risk_game.py     # Main risk game implementation
│   ├── data-analyze/    # Analysis scripts
│   │   ├── CRRA.py
│   │   ├── fit_cara.py
│   │   ├── fit_crra.py
│   │   ├── plot_crra_utility.py
│   │   └── generate_paper_results.py
│   ├── results/         # Experimental results
│   │   ├── neutral/    # Results for neutral persona
│   │   └── persona/    # Results for different personas
│   └── tools/          # Utility scripts
│       ├── estimate_epsilon.py
│       └── estimate_epsilon_opportunity_hunter.py
└── st-petersburg-games/  # St. Petersburg paradox tests
    ├── st_games.py      # Game implementation
    ├── justification.py # Reasoning analysis
    └── analyze/         # Analysis tools
        ├── analyze.py
        ├── analyze_breakpoint_decisions.py
        ├── analyze_loss_aversion.py
        └── analyze_understanding_gap.py
```


### Results Structure

```
results/
├── neutral/           # Standard decision-making results
│   ├── analysis/     # Processed analysis files
│   │   ├── all_parameters_combined.png
│   │   ├── comprehensive_analysis.png
│   │   ├── model_comparison_heatmap.png
│   │   ├── bootstrap_analysis_final.csv
│   │   └── ε_bootstrap_summary.csv
│   └── json/         # Raw experimental data
└── persona/          # Persona-based experiments
    ├── analysis/     # Analyzed results for personas
    └── json/         # Raw data including opportunity hunter results
```

## Key Findings

The analysis focuses on various aspects of LLM decision-making:

- Risk aversion parameters (r, α)
- Choice sensitivity (β)
- Ambiguity preferences (ε)
- Behavioral consistency across contexts
- Persona-based decision variations
- Model-specific behavioral patterns

### Key Visualizations

- Model comparison heatmaps
- Parameter distribution analysis
- Comprehensive behavioral analysis
- Bootstrap statistical analysis

## Dependencies

- Python 3.8+
- OpenAI Azure API
- NumPy
- Matplotlib
- Pandas
- SciPy

## License

Please contact the authors for licensing information.

## Citation

If you use this codebase in your research, please cite our work appropriately.

## Contact

For questions and feedback, please open an issue in the repository.
