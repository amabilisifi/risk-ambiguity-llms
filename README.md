# Risk and Ambiguity Analysis for LLMs

This repository provides tools and frameworks for analyzing how Large Language Models (LLMs) handle risk and ambiguity in decision-making tasks. The project includes experimental setups, analysis pipelines, and visualization tools to compare LLM behavior with human decision-making patterns.

## Project Structure

```
.
├── ambiguity-games/         # Ambiguity preference experiments
├── risk-games/             # Risk preference analysis
│   ├── data-analyze/      # Analysis scripts
│   ├── results/           # Experimental results
│   └── tools/            # Utility scripts
└── st-petersburg-games/    # St. Petersburg paradox tests
```

## Key Features

- Risk preference measurement using CRRA (Constant Relative Risk Aversion) and CARA (Constant Absolute Risk Aversion) models
- Ambiguity aversion analysis with ε-contamination framework
- Opportunity hunter persona experiments
- Comprehensive visualization and statistical analysis tools
- Automated experiment pipelines for multiple LLM models

## Getting Started

1. Install dependencies:

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

## Key Results

The analysis focuses on various aspects of LLM decision-making:

- Risk aversion parameters (r, α)
- Choice sensitivity (β)
- Ambiguity preferences (ε)
- Behavioral consistency across different contexts

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
