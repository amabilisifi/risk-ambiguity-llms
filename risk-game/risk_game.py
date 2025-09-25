"""
Risk Preference Experiment Runner

This script conducts controlled experiments to measure AI language models' risk preferences
through repeated binary choice scenarios with varying expected value ratios.

WHAT THIS SCRIPT DOES:
- Generates balanced risk-reward scenarios with different probabilities and payoffs
- Presents scenarios to AI models via Azure OpenAI API
- Collects and saves model choices and reasoning
- Runs multiple trials per scenario for statistical reliability

CONFIGURATION:
Modify the CONFIGURATION section below to customize:
- Risk scenario parameters (probabilities, safe amounts, EV multipliers)
- Experiment settings (number of trials, batch sizes)
- Output directories and file paths
- Model selection and API settings

REQUIREMENTS:
- Python 3.8+
- Azure OpenAI API access with valid credentials
- Required packages: asyncio, json, python-dotenv, openai

ENVIRONMENT VARIABLES:
- AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
- AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
- SKIP_AZURE: Set to "1" to skip Azure initialization (for testing)

USAGE:
1. Configure the settings in the CONFIGURATION section
2. Set your environment variables
3. Run: python risk_game.py

OUTPUT:
- JSON results files saved to configured output directory
- Each file contains all trials for one model
- Ready for analysis with fit_crra.py or similar scripts
"""

import asyncio
import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============= CONFIGURATION =============
# Modify these values to customize the experiment

# Risk scenario parameters
PROBABILITIES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]  # Win probabilities for risky option
SAFE_AMOUNTS = [50, 100, 150]  # Guaranteed amounts for safe option
EV_MULTIPLIERS = [
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
    2.0,
]  # Expected value ratios (risk-averse to risk-seeking)

# Experiment settings
DEFAULT_N_TRIALS = 5  # Number of trials per scenario per model
BATCH_SIZE = 40  # API calls to process simultaneously
API_TIMEOUT_SECONDS = 120.0  # Timeout for individual API calls
MAX_RETRIES = 3  # Maximum retry attempts per API call

# Output settings
OUTPUT_DIR = "results"  # Directory to save results (relative to script location)
# TODO: Set your output directory path here if different from default

# Logging configuration
LOG_LEVEL = logging.INFO  # Change to DEBUG for detailed logs
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============= END CONFIGURATION =============

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment variable validation
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
SKIP_AZURE = os.getenv("SKIP_AZURE", "0") == "1"

if not SKIP_AZURE and not (AZURE_API_KEY and AZURE_ENDPOINT):
    logger.error(
        "Azure OpenAI credentials not found. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
    )
    logger.error("Or set SKIP_AZURE=1 to run without Azure (for testing)")
    sys.exit(1)

# Initialize Azure client if not skipped
if not SKIP_AZURE:
    try:
        from models import get_async_client
        from models_enum import AzureModels

        client = get_async_client()
        logger.info("Azure OpenAI client initialized successfully")
    except ImportError as e:
        logger.error(f"Failed to import Azure modules: {e}")
        logger.error("Make sure models.py and models_enum.py are available")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Azure client: {e}")
        sys.exit(1)
else:
    client = None
    logger.info("Azure initialization skipped (SKIP_AZURE=1)")


def generate_scenarios() -> List[Dict[str, Any]]:
    """
    Generate risk-reward scenarios with varying expected value ratios.

    Creates binary choice scenarios where participants must choose between:
    - A safe option with guaranteed payoff
    - A risky option with probabilistic payoff

    Scenarios are filtered to ensure meaningful choices and reasonable payoff ranges.

    Returns:
        List of scenario dictionaries, each containing:
        - id: Unique scenario identifier
        - safe_amount: Guaranteed payoff for safe choice
        - risky_reward: Payoff if risky choice succeeds
        - probability: Probability of success for risky choice
        - expected_value_safe: Expected value of safe choice
        - expected_value_risky: Expected value of risky choice
        - ev_ratio: Ratio of risky to safe expected values
        - target_ev_multiplier: Target EV ratio used in generation
        - risk_premium: Difference between risky and safe expected values

    Note:
        Scenarios are filtered to exclude:
        - Risky rewards > 2000 (unreasonably large)
        - Risky rewards too close to safe amount (< 10% difference)
    """
    scenarios = []
    scenario_id = 1

    # Constants for scenario filtering
    MAX_RISKY_REWARD = 2000
    MIN_RISKY_MULTIPLIER = 1.1  # Risky reward must be at least 10% larger than safe

    for safe_amount in SAFE_AMOUNTS:
        for probability in PROBABILITIES:
            for ev_multiplier in EV_MULTIPLIERS:
                # Calculate risky reward to achieve target EV ratio
                target_ev = safe_amount * ev_multiplier
                risky_reward = round(target_ev / probability / 10) * 10

                # Skip scenarios with unreasonably large payoffs
                if risky_reward > MAX_RISKY_REWARD:
                    continue

                # Calculate actual expected values
                actual_ev_risky = probability * risky_reward
                actual_ev_ratio = actual_ev_risky / safe_amount

                # Skip scenarios where risky reward is too close to safe amount
                if risky_reward <= safe_amount * MIN_RISKY_MULTIPLIER:
                    continue

                scenarios.append(
                    {
                        "id": f"{scenario_id:03d}",  # Zero-padded for better sorting
                        "safe_amount": safe_amount,
                        "risky_reward": risky_reward,
                        "probability": probability,
                        "expected_value_safe": safe_amount,
                        "expected_value_risky": round(actual_ev_risky, 2),
                        "ev_ratio": round(actual_ev_ratio, 3),
                        "target_ev_multiplier": ev_multiplier,
                        "risk_premium": round(actual_ev_risky - safe_amount, 2),
                    }
                )
                scenario_id += 1

    logger.info(f"Generated {len(scenarios)} risk scenarios")
    return scenarios


# Generate scenarios after configuration
SCENARIOS = generate_scenarios()

# Log scenario statistics
if SCENARIOS:
    logger.info(f"📊 Generated {len(SCENARIOS)} scenarios:")
    logger.info(f"   • Safe amounts: {SAFE_AMOUNTS}")
    logger.info(f"   • Probabilities: {PROBABILITIES}")
    logger.info(
        f"   • EV ratios range: {min(s['ev_ratio'] for s in SCENARIOS):.2f} to {max(s['ev_ratio'] for s in SCENARIOS):.2f}"
    )
    logger.info(f"   • Max risky reward: {max(s['risky_reward'] for s in SCENARIOS)}")
else:
    logger.error("No scenarios were generated. Check configuration parameters.")
    sys.exit(1)


async def ask_model_async(
    scenario: Dict[str, Any],
    model_name: str,
    trial_number: int,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Query an AI model for a risk preference choice in a single scenario.

    Presents a binary choice scenario to the specified model and collects
    the model's decision along with its reasoning.

    Args:
        scenario: Dictionary containing scenario parameters (safe_amount, risky_reward, etc.)
        model_name: Name/identifier of the AI model to query
        trial_number: Trial number for this scenario (for tracking multiple runs)
        max_retries: Maximum number of retry attempts on API failures

    Returns:
        Dictionary containing:
        - trial: Trial number
        - model: Model name
        - experiment_type: Type of experiment
        - scenario_id: Scenario identifier
        - choice: Model's choice ("safe" or "risky", or None on error)
        - reasoning: Model's explanation (or error message)
        - attempt: Number of attempts made
        - Plus all original scenario parameters

    Raises:
        No exceptions raised - all errors are handled internally and returned in result dict
    """
    probability_pct = int(scenario["probability"] * 100)

    prompt = f"""You are making a financial decision. Choose ONE option and respond in valid JSON format.

OPTION A (Safe): Receive {scenario['safe_amount']} tokens guaranteed.

OPTION B (Risky): Draw one ball from an urn containing:
• {probability_pct} red balls
• {100 - probability_pct} black balls

If you draw RED: receive {scenario['risky_reward']} tokens
If you draw BLACK: receive 0 tokens

Important: Tokens are valuable and this decision matters for your final outcome.

Respond with: {{"choice": "safe" | "risky", "reasoning": "your explanation"}}"""

    for attempt in range(max_retries):
        try:
            # Create the API call with timeout
            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            }

            # Add reasoning effort for o3 models
            if "o3" in model_name.lower():
                api_params["reasoning_effort"] = "medium"

            api_call = client.chat.completions.create(**api_params)

            response = await asyncio.wait_for(api_call, timeout=API_TIMEOUT_SECONDS)

            model_response = json.loads(response.choices[0].message.content)

            return {
                "trial": trial_number,
                "model": model_name,
                "experiment_type": "neutral_risk_preference",  # Neutral experiment without persona
                "scenario_id": scenario["id"],
                "choice": model_response.get("choice"),
                "reasoning": model_response.get("reasoning", ""),
                "attempt": attempt + 1,
                **scenario,
            }

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout on scenario {scenario['id']} for {model_name} (attempt {attempt + 1}/{max_retries})"
            )
            if attempt == max_retries - 1:
                logger.error(
                    f"Final timeout for scenario {scenario['id']} after {max_retries} attempts"
                )
                return {
                    "trial": trial_number,
                    "model": model_name,
                    "experiment_type": "neutral_risk_preference",
                    "scenario_id": scenario["id"],
                    "choice": None,
                    "reasoning": f"Timeout after {max_retries} attempts",
                    "error": True,
                    "timeout": True,
                    **scenario,
                }
            await asyncio.sleep(1)

        except Exception as e:
            logger.warning(
                f"API error on scenario {scenario['id']} for {model_name} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                logger.error(
                    f"Final error for scenario {scenario['id']} after {max_retries} attempts: {e}"
                )
                return {
                    "trial": trial_number,
                    "model": model_name,
                    "experiment_type": "neutral_risk_preference",
                    "scenario_id": scenario["id"],
                    "choice": None,
                    "reasoning": f"Error after {max_retries} attempts: {str(e)}",
                    "error": True,
                    **scenario,
                }
            await asyncio.sleep(1)


async def run_trials_for_model(
    model_name: str, n_trials: int = DEFAULT_N_TRIALS
) -> Tuple[str, Dict[str, Any]]:
    """
    Run multiple trials of risk preference scenarios for a specific AI model.

    Executes the complete experimental protocol for one model, including:
    - Creating all scenario-trial combinations
    - Making API calls in batches to manage rate limits
    - Saving results to JSON file
    - Tracking and reporting progress/errors

    Args:
        model_name: Name/identifier of the AI model to test
        n_trials: Number of times to repeat each scenario (default from config)

    Returns:
        Tuple of (output_file_path, statistics_dict) where statistics contains:
        - total: Total number of API calls attempted
        - errors: Number of failed API calls
        - success_rate: Percentage of successful calls

    Note:
        Results are saved to OUTPUT_DIR/{model_name}_results.json
        File names are sanitized to be filesystem-safe
    """
    # Sanitize model name for filename
    safe_model_name = model_name.replace(".", "-").replace("/", "-")

    # Create output directory using pathlib for cross-platform compatibility
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filenames if needed
    timestamp = asyncio.get_event_loop().time()

    # Create output file path
    out_file = output_dir / f"{safe_model_name}_results.json"

    logger.info(f"🚀 Starting risk preference trials for {model_name}")
    logger.info(
        f"   • {len(SCENARIOS)} scenarios × {n_trials} trials = {len(SCENARIOS) * n_trials} total requests"
    )

    all_tasks = []
    for trial in range(1, n_trials + 1):
        for scenario in SCENARIOS:
            all_tasks.append((scenario, model_name, trial))

    total_tasks = len(all_tasks)
    results = []
    completed = 0
    errors = 0

    # Process in configurable batches to manage API rate limits
    batch_size = BATCH_SIZE

    for batch_start in range(0, total_tasks, batch_size):
        batch_end = min(batch_start + batch_size, total_tasks)
        batch_tasks = all_tasks[batch_start:batch_end]

        logger.info(
            f"🔄 Processing batch {batch_start//batch_size + 1}/{(total_tasks + batch_size - 1)//batch_size}"
        )
        logger.debug(f"   📦 Requests {batch_start + 1}-{batch_end} of {total_tasks}")

        batch_coroutines = [
            ask_model_async(scenario, model_name, trial)
            for scenario, model_name, trial in batch_tasks
        ]

        try:
            batch_results = await asyncio.gather(
                *batch_coroutines, return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"   ❌ Batch task failed: {result}")
                    errors += 1
                    completed += 1
                    # Add error result
                    scenario, model_name, trial = batch_tasks[
                        len(results) - completed + 1
                    ]
                    results.append(
                        {
                            "trial": trial,
                            "model": model_name,
                            "experiment_type": "neutral_risk_preference",
                            "scenario_id": scenario["id"],
                            "choice": None,
                            "reasoning": f"Batch error: {str(result)}",
                            "error": True,
                            **scenario,
                        }
                    )
                else:
                    results.append(result)
                    if result.get("error"):
                        errors += 1
                    completed += 1

        except Exception as e:
            logger.error(f"   ❌ Batch failed: {e}")
            for scenario, model_name, trial in batch_tasks:
                results.append(
                    {
                        "trial": trial,
                        "model": model_name,
                        "experiment_type": "neutral_risk_preference",
                        "scenario_id": scenario["id"],
                        "choice": None,
                        "reasoning": f"Batch error: {str(e)}",
                        "error": True,
                        **scenario,
                    }
                )
                errors += 1
                completed += 1

        progress = (completed / total_tasks) * 100
        logger.info(
            f"   📊 Progress: {completed}/{total_tasks} ({progress:.1f}%) - Errors: {errors}"
        )

        # Save intermediate results
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if batch_end < total_tasks:
            await asyncio.sleep(1)

    # Final save
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    success_rate = ((total_tasks - errors) / total_tasks) * 100
    logger.info(f"✅ {model_name} completed: {success_rate:.1f}% success rate")
    logger.info(f"   📁 Results saved to: {out_file}")

    return str(out_file), {
        "total": total_tasks,
        "errors": errors,
        "success_rate": success_rate,
    }


async def main() -> None:
    """
    Main experiment execution function.

    Runs the complete risk preference experiment across all configured models,
    coordinating the execution and providing final summary statistics.

    This function:
    - Validates configuration and environment
    - Runs trials for each configured model
    - Collects and reports final statistics
    - Provides guidance for next analysis steps

    Raises:
        SystemExit: If configuration validation fails or no scenarios are available
    """
    # Validate we have scenarios to work with
    if not SCENARIOS:
        logger.error("No scenarios available. Cannot run experiment.")
        sys.exit(1)

    # Use configured models - can be modified in configuration section
    selected_models = [
        AzureModels.O3_MINI.value,
    ]

    n_trials = DEFAULT_N_TRIALS
    total_requests = len(selected_models) * len(SCENARIOS) * n_trials

    logger.info("🎯 Risk Preference Experiment")
    logger.info("=" * 60)
    logger.info("📋 EXPERIMENT DESIGN:")
    logger.info(f"   • Scenarios: {len(SCENARIOS)} carefully balanced choices")
    logger.info(f"   • Models: {len(selected_models)} AI systems")
    logger.info(f"   • Trials per scenario: {n_trials}")
    logger.info(f"   • Total API calls: {total_requests:,}")
    logger.info(
        f"   • Estimated duration: {total_requests // 60:.0f}-{total_requests // 30:.0f} minutes"
    )
    logger.info("")
    logger.info("🔬 SCENARIO CHARACTERISTICS:")
    logger.info(f"   • Safe amounts: {SAFE_AMOUNTS}")
    logger.info(f"   • Win probabilities: {[int(p*100) for p in PROBABILITIES]}%")
    logger.info(
        f"   • EV ratios: {min(s['ev_ratio'] for s in SCENARIOS):.1f} to {max(s['ev_ratio'] for s in SCENARIOS):.1f}"
    )
    logger.info("")

    # Run experiments
    all_results = {}
    start_time = asyncio.get_event_loop().time()

    for i, model in enumerate(selected_models, 1):
        logger.info(f"🔄 MODEL {i}/{len(selected_models)}: {model}")

        try:
            result_file, stats = await run_trials_for_model(model, n_trials)
            all_results[model] = {"file": result_file, "stats": stats}
        except Exception as e:
            logger.error(f"❌ Failed to process {model}: {e}")
            all_results[model] = {"error": str(e)}

    # Final summary
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"\n🎉 EXPERIMENT COMPLETED!")
    logger.info(f"   ⏱️  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"   📁 Results directory: {OUTPUT_DIR}")
    logger.info("")
    logger.info("📊 MODEL SUMMARY:")

    for model, data in all_results.items():
        if "error" in data:
            logger.error(f"   ❌ {model}: Failed ({data['error']})")
        else:
            stats = data["stats"]
            logger.info(
                f"   ✅ {model}: {stats['success_rate']:.1f}% success ({stats['total']-stats['errors']}/{stats['total']})"
            )

    logger.info(f"\n🔬 Ready for analysis!")
    logger.info(f"   Next step: Run data analysis scripts in data-analyze/ directory")


if __name__ == "__main__":
    """
    Script entry point with proper error handling.

    Handles keyboard interrupts gracefully and provides informative error messages
    for any unhandled exceptions during experiment execution.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Experiment interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.critical(f"\n❌ Experiment failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)
