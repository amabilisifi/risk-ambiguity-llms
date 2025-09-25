"""
St. Petersburg Paradox Experiment for AI Language Models

This script investigates how AI language models respond to the famous St. Petersburg Paradox,
a classic decision-making problem with theoretically infinite expected value but finite
willingness to pay.

THE ST. PETERSBURG PARADOX:
The game starts with $2 in the pot. A coin is flipped repeatedly until heads appears.
Each tails doubles the pot (2→4→8→16...). Heads ends the game and you win the pot amount.
Expected value is infinite: E[X] = Σ(1/2^k * 2^k) = Σ(1) = ∞ for k=1 to ∞

WHAT THIS SCRIPT DOES:
- Tests AI willingness to pay various entry fees for the St. Petersburg game
- Uses multiple trials per price point for statistical reliability
- Compares behavior across different AI models
- Generates comprehensive results with decision analysis
- Supports resume functionality for interrupted experiments

PROCEDURE:
1. Test entry prices from $1 to $100,000 (10 price points)
2. Run multiple trials per price per model
3. Record PLAY/PASS decisions and analyze willingness-to-pay
4. Save results with detailed metadata and statistics

THEORETICAL SIGNIFICANCE:
The paradox demonstrates the difference between mathematical expectation and
psychological willingness to pay. AI models may show finite willingness-to-pay
despite infinite expected value, revealing bounded rationality.

REQUIREMENTS:
- Python 3.8+
- Azure OpenAI API access with valid credentials
- Required packages: asyncio, json, python-dotenv, openai

USAGE:
1. Run from st-petersburg-games/ directory
2. Default: python st_games.py (uses configured models and settings)
3. Results saved to results/{model_name}/ directories
4. Experiment resumes automatically if interrupted

OUTPUT:
- Individual JSON results for each model
- Comprehensive statistics and price-by-price breakdown
- Resume capability via partial result files
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from models import get_async_client
from models_enum import AzureModels

# Import persona prompt if available
try:
    from risk_taking_persona import SYSTEM_PROMPT

    PERSONA_AVAILABLE = True
except ImportError:
    SYSTEM_PROMPT = ""
    PERSONA_AVAILABLE = False

# ============= CONFIGURATION =============
# Modify these values to customize the experiment

# Model selection
SELECTED_MODELS = [
    # AzureModels.O3_MINI.value,  # o3-mini with medium reasoning
    AzureModels.GPT_5.value,  # gpt-5
    AzureModels.GPT_4O_MINI.value,  # 4o-mini
    AzureModels.GPT_4O.value,  # 4o
    AzureModels.GPT_4_1.value,  # 4.1
]
# TODO: Uncomment/add models as needed for your experiments

# Experiment parameters
N_TRIALS_PER_PRICE = 10  # Number of trials per entry price per model
TIMEOUT_SECONDS = 180  # Timeout per API call in seconds
MAX_RETRIES = 3  # Maximum retry attempts per trial

# Directory configuration
RESULTS_DIR = "results"  # Base directory for all results
# TODO: Set your results directory path here if different from default

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============= END CONFIGURATION =============

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load environment variables and initialize client
load_dotenv()

try:
    client = get_async_client()
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure client: {e}")
    logger.error("Make sure Azure credentials are properly configured")
    sys.exit(1)

# Create results directories for each model
results_path = Path(RESULTS_DIR)
results_path.mkdir(exist_ok=True)

for model in SELECTED_MODELS:
    safe_model_name = model.replace(".", "-").replace("/", "-")
    model_dir = results_path / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created results directory: {model_dir}")

# St. Petersburg Paradox Entry Prices Configuration
ENTRY_PRICES = [
    {"id": "price_01", "entry_fee": 1},  # $1
    {"id": "price_02", "entry_fee": 2},  # $2
    {"id": "price_03", "entry_fee": 4},  # $4
    {"id": "price_04", "entry_fee": 8},  # $8
    {"id": "price_05", "entry_fee": 16},  # $16
    {"id": "price_06", "entry_fee": 32},  # $32
    {"id": "price_07", "entry_fee": 100},  # $100
    {"id": "price_08", "entry_fee": 1000},  # $1,000
    {"id": "price_09", "entry_fee": 10000},  # $10,000
    {"id": "price_10", "entry_fee": 100000},  # $100,000
]


def create_st_petersburg_prompt(entry_fee: int) -> str:
    """
    Create the St. Petersburg Paradox game prompt for AI models.

    Generates a comprehensive prompt explaining the game rules, possible outcomes,
    and requiring a PLAY/PASS decision in JSON format.

    Args:
        entry_fee: The dollar amount required to enter the game

    Returns:
        Formatted prompt string for the AI model
    """
    return f"""
You are offered to play the famous "St. Petersburg Paradox" game. Here are the COMPLETE rules:

**GAME RULES:**
1. The game starts with $2 in the pot
2. A fair coin is flipped repeatedly until it lands HEADS for the first time
3. Each time the coin lands TAILS, the pot doubles (2 → 4 → 8 → 16 → 32...)
4. When the coin finally lands HEADS, the game ends and you win whatever amount is in the pot
5. The coin could potentially be flipped INFINITE times if it keeps landing tails

**POSSIBLE OUTCOMES & PAYOFFS:**
- Heads on 1st flip: Win $2 (probability = 1/2)
- Tails then Heads: Win $4 (probability = 1/4)
- Tails-Tails then Heads: Win $8 (probability = 1/8)
- Tails-Tails-Tails then Heads: Win $16 (probability = 1/16)
- And so on... potentially forever

**YOUR DECISION:**
The entry fee to play this game is ${entry_fee:,}.

You must decide: Will you pay ${entry_fee:,} to play this game, or will you pass?

**REQUIRED RESPONSE FORMAT:**
You must respond ONLY in valid JSON format with exactly this field:
{{
    "decision": "PLAY" | "PASS"
}}

Make your decision and respond with only PLAY or PASS in the JSON format above.
"""


async def ask_model_async(
    price_case: Dict[str, Any],
    trial_number: int,
    model_name: str,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Query AI model for St. Petersburg Paradox decision with retry logic.

    Makes an API call to the specified model asking whether to play the St. Petersburg
    game at the given entry fee. Includes comprehensive error handling and retry logic.

    Args:
        price_case: Dictionary with price case info (id, entry_fee)
        trial_number: Trial number for tracking (1-based)
        model_name: Name/identifier of the AI model to query
        max_retries: Maximum number of retry attempts on failures

    Returns:
        Dictionary containing the decision result with metadata:
        - price_id: Price case identifier
        - entry_fee: Entry fee amount
        - trial_number: Trial number
        - timestamp: ISO timestamp of the decision
        - Decision: "PLAY", "PASS", or "ERROR"
        - Plus additional metadata fields
    """
    for attempt in range(max_retries):
        try:
            # Prepare API parameters with persona integration
            messages = []

            # Add persona prompt if available
            if PERSONA_AVAILABLE and SYSTEM_PROMPT:
                messages.append(
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    }
                )

            # Add experiment context
            messages.append(
                {
                    "role": "system",
                    "content": "You are participating in a decision-making experiment about the St. Petersburg Paradox. Respond only with valid JSON containing your decision: either PLAY or PASS.",
                }
            )

            # Add the game prompt
            messages.append(
                {
                    "role": "user",
                    "content": create_st_petersburg_prompt(price_case["entry_fee"]),
                }
            )

            api_params = {
                "model": model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }

            # Make the API call with timeout
            api_call = client.chat.completions.create(**api_params)
            response = await asyncio.wait_for(api_call, timeout=TIMEOUT_SECONDS)

            model_response = json.loads(response.choices[0].message.content)

            # Extract and validate decision
            decision = model_response.get("decision", "ERROR").upper()
            if decision not in ["PLAY", "PASS"]:
                decision = "ERROR"

            # Return successful result with metadata
            return {
                "price_id": price_case["id"],
                "entry_fee": price_case["entry_fee"],
                "trial_number": trial_number,
                "timestamp": datetime.now().isoformat(),
                "attempt": attempt + 1,  # Track which attempt succeeded
                "Decision": decision,
                "Entrance Fee": price_case["entry_fee"],
            }

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout on trial {trial_number} for ${price_case['entry_fee']:,} "
                f"({model_name}, attempt {attempt + 1}/{max_retries})"
            )
            if attempt == max_retries - 1:  # Last attempt failed
                logger.error(
                    f"Final timeout for trial {trial_number} at ${price_case['entry_fee']:,}"
                )
                return {
                    "price_id": price_case["id"],
                    "entry_fee": price_case["entry_fee"],
                    "trial_number": trial_number,
                    "timestamp": datetime.now().isoformat(),
                    "Entrance Fee": price_case["entry_fee"],
                    "Decision": "ERROR",
                    "error": True,
                    "timeout": True,
                }
            await asyncio.sleep(1)

        except Exception as e:
            logger.warning(
                f"API error in trial {trial_number} for ${price_case['entry_fee']:,} "
                f"({model_name}, attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:  # Last attempt failed
                logger.error(
                    f"Final error for trial {trial_number} at ${price_case['entry_fee']:,}: {e}"
                )
                return {
                    "price_id": price_case["id"],
                    "entry_fee": price_case["entry_fee"],
                    "trial_number": trial_number,
                    "timestamp": datetime.now().isoformat(),
                    "Entrance Fee": price_case["entry_fee"],
                    "Decision": "ERROR",
                    "error": True,
                }
            await asyncio.sleep(1)

    # This should never be reached, but just in case
    logger.error(f"Unexpected end of retry loop for trial {trial_number}")
    return {
        "price_id": price_case["id"],
        "entry_fee": price_case["entry_fee"],
        "trial_number": trial_number,
        "timestamp": datetime.now().isoformat(),
        "Entrance Fee": price_case["entry_fee"],
        "Decision": "ERROR",
        "error": True,
    }


def load_partial_results(
    price_case: Dict[str, Any], model_name: str
) -> List[Dict[str, Any]]:
    """
    Load existing partial results for a specific price case and model.

    Supports resume functionality by loading previously completed trials
    for a given price case, allowing experiments to continue from where
    they left off if interrupted.

    Args:
        price_case: Dictionary with price case info (id, entry_fee)
        model_name: Name of the AI model

    Returns:
        List of previously completed trial results, or empty list if none found
    """
    safe_model_name = model_name.replace(".", "-").replace("/", "-")
    model_dir = results_path / safe_model_name
    partial_file = model_dir / f"partial_{price_case['id']}.json"

    if partial_file.exists():
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load partial results file: {partial_file}")
            return []
    return []


def save_partial_results(
    price_case: Dict[str, Any], results: List[Dict[str, Any]], model_name: str
) -> None:
    """
    Save partial results for resume functionality.

    Saves the current state of trial results for a price case, allowing
    the experiment to resume from this point if interrupted.

    Args:
        price_case: Dictionary with price case info (id, entry_fee)
        results: List of trial results to save
        model_name: Name of the AI model
    """
    safe_model_name = model_name.replace(".", "-").replace("/", "-")
    model_dir = results_path / safe_model_name
    partial_file = model_dir / f"partial_{price_case['id']}.json"

    with open(partial_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved partial results: {partial_file}")


async def run_price_trials_with_timeout(
    price_case: Dict[str, Any], model_name: str
) -> List[Dict[str, Any]]:
    """
    Run multiple trials for a specific entry price with timeout and resume support.

    Executes the required number of trials for a given entry price, with support
    for resuming interrupted experiments by loading partial results.

    Args:
        price_case: Dictionary with price case info (id, entry_fee)
        model_name: Name of the AI model to test

    Returns:
        List of all trial results (existing + newly completed)
    """
    logger.info(
        f"Starting trials for entry price: ${price_case['entry_fee']:,} ({model_name})"
    )

    # Load any existing results for resume functionality
    existing_results = load_partial_results(price_case, model_name)
    completed_trials = {r["trial_number"] for r in existing_results}

    if len(completed_trials) == N_TRIALS_PER_PRICE:
        logger.info(
            f"✅ All {N_TRIALS_PER_PRICE} trials already completed for ${price_case['entry_fee']:,}"
        )
        return existing_results

    # Find missing trials
    missing_trials = [
        t for t in range(1, N_TRIALS_PER_PRICE + 1) if t not in completed_trials
    ]

    if completed_trials:
        logger.info(
            f"📋 Found {len(completed_trials)} existing results, "
            f"running {len(missing_trials)} missing trials"
        )
        logger.debug(f"Missing trials: {missing_trials}")
    else:
        logger.info(f"🆕 Running all {N_TRIALS_PER_PRICE} trials")

    # Create tasks for missing trials only
    logger.info(
        f"⏱️  Starting {len(missing_trials)} requests with {TIMEOUT_SECONDS}s timeout..."
    )

    successful_results: List[Dict[str, Any]] = []
    retry_trials = missing_trials.copy()

    for retry_attempt in range(MAX_RETRIES):
        if not retry_trials:
            break

        logger.info(
            f"🔄 Attempt {retry_attempt + 1}/{MAX_RETRIES}: Processing {len(retry_trials)} trials"
        )

        # Create fresh tasks for current retry
        tasks = [
            ask_model_async(price_case, trial, model_name) for trial in retry_trials
        ]

        try:
            # Run current batch with timeout
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            completed_trials = []
            failed_trials = []

            for i, result in enumerate(batch_results):
                trial_num = retry_trials[i]

                if isinstance(result, Exception):
                    logger.error(f"   ❌ Trial {trial_num} failed: {str(result)}")
                    failed_trials.append(trial_num)
                else:
                    logger.debug(f"   ✅ Trial {trial_num} completed")
                    successful_results.append(result)
                    completed_trials.append(trial_num)

            # Update retry list to only failed trials
            retry_trials = failed_trials

            logger.info(
                f"   📊 Batch {retry_attempt + 1}: {len(completed_trials)} completed, {len(failed_trials)} failed"
            )

            # Save progress after each batch
            all_results = existing_results + successful_results
            save_partial_results(price_case, all_results, model_name)

        except Exception as e:
            logger.error(f"   ❌ Batch {retry_attempt + 1} failed completely: {e}")
            # Continue to next retry attempt

    # Create error results for any remaining failed trials
    for trial_num in retry_trials:
        error_result = {
            "price_id": price_case["id"],
            "entry_fee": price_case["entry_fee"],
            "trial_number": trial_num,
            "timestamp": datetime.now().isoformat(),
            "Entrance Fee": price_case["entry_fee"],
            "Decision": "ERROR",
        }
        successful_results.append(error_result)

    # Final combine and save
    all_results = existing_results + successful_results
    save_partial_results(price_case, all_results, model_name)

    successful_count = len(
        [r for r in successful_results if r.get("Decision") != "ERROR"]
    )
    logger.info(
        f"✅ Completed {len(successful_results)} trials for ${price_case['entry_fee']:,} "
        f"({successful_count} successful)"
    )
    return all_results


def find_resume_point(model_name: str) -> int:
    """Find which price to resume from based on existing partial files for this model"""
    for i, price_case in enumerate(ENTRY_PRICES):
        existing_results = load_partial_results(price_case, model_name)
        if len(existing_results) < N_TRIALS_PER_PRICE:
            print(f"🔄 Resuming from price ${price_case['entry_fee']:,} (index {i})")
            return i

    print(f"✅ All prices appear complete, starting from beginning")
    return 0


async def run_experiment_for_model(model_name: str) -> dict:
    """Run the complete St. Petersburg Paradox experiment for a specific model"""

    print(f"\n🚀 Running experiment for model: {model_name}")
    print("=" * 60)

    # Check for resume point for this model
    resume_index = find_resume_point(model_name)

    model_results = []

    # Start from resume point
    for i in range(resume_index, len(ENTRY_PRICES)):
        price_case = ENTRY_PRICES[i]

        print(
            f"\n[{i+1}/{len(ENTRY_PRICES)}] Processing entry fee: ${price_case['entry_fee']:,}"
        )

        # Run trials with timeout and resume support
        price_results = await run_price_trials_with_timeout(price_case, model_name)
        model_results.extend(price_results)

        # Quick summary for this price
        play_count = sum(1 for r in price_results if r.get("Decision") == "PLAY")
        pass_count = sum(1 for r in price_results if r.get("Decision") == "PASS")
        error_count = sum(1 for r in price_results if r.get("Decision") == "ERROR")

        print(
            f"   📊 Results: {play_count} PLAY, {pass_count} PASS, {error_count} ERRORS"
        )

        # Show individual choices
        choices = [r.get("Decision", "ERROR") for r in price_results]
        print(f"   📋 Choices: {choices}")

    return model_results


async def main() -> None:
    """
    Run the complete St. Petersburg Paradox experiment with separate results per model.

    Executes the full experimental protocol across all configured models,
    testing willingness to pay across different entry prices with comprehensive
    error handling, resume capability, and detailed result storage.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("🎯 Starting Multi-Model St. Petersburg Paradox Experiment")
    logger.info(f"🤖 Models: {len(SELECTED_MODELS)} AI systems")
    logger.info(f"📊 Testing {len(ENTRY_PRICES)} different entry prices per model")
    logger.info(f"🔄 Running {N_TRIALS_PER_PRICE} trials per price per model")
    logger.info(f"⏱️  Timeout: {TIMEOUT_SECONDS} seconds per price")

    total_trials = len(SELECTED_MODELS) * len(ENTRY_PRICES) * N_TRIALS_PER_PRICE
    logger.info(f"📈 Total trials across all models: {total_trials:,}")

    logger.info("\n🤖 SELECTED MODELS:")
    for i, model in enumerate(SELECTED_MODELS, 1):
        logger.info(f"   {i}. {model}")

    logger.info("-" * 60)

    # Run experiment for each model separately
    for model_idx, model_name in enumerate(SELECTED_MODELS, 1):
        logger.info(f"\n🎯 MODEL {model_idx}/{len(SELECTED_MODELS)}: {model_name}")

        try:
            # Run experiment for this model
            model_results = await run_experiment_for_model(model_name)

            # Save individual model results
            safe_model_name = model_name.replace(".", "-").replace("/", "-")
            model_dir = results_path / safe_model_name
            ts_model = datetime.now().strftime("%Y%m%d_%H%M%S")
            outfile = (
                model_dir / f"st_petersburg_final_{safe_model_name}_{ts_model}.json"
            )

            # Calculate model statistics
            play_count = sum(1 for r in model_results if r.get("Decision") == "PLAY")
            pass_count = sum(1 for r in model_results if r.get("Decision") == "PASS")
            error_count = sum(1 for r in model_results if r.get("Decision") == "ERROR")
            total_count = len(model_results)
            play_percentage = (
                round((play_count / total_count * 100), 2) if total_count > 0 else 0
            )

            # Create individual model results
            model_experiment_results = {
                "experiment_info": {
                    "experiment_name": f"St. Petersburg Paradox - {model_name}",
                    "model_tested": model_name,
                    "total_trials": len(model_results),
                    "trials_per_price": N_TRIALS_PER_PRICE,
                    "entry_prices_tested": [p["entry_fee"] for p in ENTRY_PRICES],
                    "experiment_date": datetime.now().isoformat(),
                    "expected_value": "INFINITE (mathematical)",
                    "temperature": 0,
                    "timeout_seconds": TIMEOUT_SECONDS,
                },
                "summary_statistics": {
                    "total_play_decisions": play_count,
                    "total_pass_decisions": pass_count,
                    "total_errors": error_count,
                    "play_percentage": play_percentage,
                    "by_price_breakdown": {},
                },
                "detailed_results": model_results,
            }

            # Add price-by-price breakdown for this model
            for price_case in ENTRY_PRICES:
                price_results = [
                    r for r in model_results if r.get("price_id") == price_case["id"]
                ]
                price_play_count = sum(
                    1 for r in price_results if r.get("Decision") == "PLAY"
                )
                price_pass_count = sum(
                    1 for r in price_results if r.get("Decision") == "PASS"
                )

                model_experiment_results["summary_statistics"]["by_price_breakdown"][
                    f"${price_case['entry_fee']:,}"
                ] = {
                    "play_count": price_play_count,
                    "pass_count": price_pass_count,
                    "play_percentage": (
                        round(price_play_count / len(price_results) * 100, 2)
                        if price_results
                        else 0
                    ),
                    "total_trials": len(price_results),
                }

            # Save individual model results
            with open(outfile, "w", encoding="utf-8") as f:
                json.dump(model_experiment_results, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Completed {model_name}: {len(model_results)} total trials")
            logger.info(f"💾 Results saved to: {outfile}")

        except Exception as e:
            logger.error(f"❌ Failed to process {model_name}: {e}")
            continue

    logger.info("\n" + "=" * 80)
    logger.info("🎉 MULTI-MODEL EXPERIMENT COMPLETED!")
    logger.info(f"🤖 Models tested: {len(SELECTED_MODELS)}")
    logger.info(f"📁 Each model's results saved to: {RESULTS_DIR}/{{model_name}}/")

    # Clean up partial files after successful completion
    logger.info("🧹 Cleaning up partial result files...")
    for model_name in SELECTED_MODELS:
        safe_model_name = model_name.replace(".", "-").replace("/", "-")
        model_dir = results_path / safe_model_name
        for price_case in ENTRY_PRICES:
            partial_file = model_dir / f"partial_{price_case['id']}.json"
            if partial_file.exists():
                partial_file.unlink()
                logger.debug(f"Removed partial file: {partial_file}")

    logger.info("✅ All partial files cleaned up!")
    logger.info("=" * 80)


def main_entry() -> None:
    """
    Main entry point for the St. Petersburg Paradox experiment.

    Provides user-friendly startup information and handles the async main execution
    with proper error handling for keyboard interrupts and other exceptions.
    """
    logger.info("🚀 Starting Multi-Model St. Petersburg Paradox Experiment")
    logger.info(
        f"🤖 Testing {len(SELECTED_MODELS)} models: {', '.join(SELECTED_MODELS)}"
    )
    logger.info("💡 To change models, modify SELECTED_MODELS in configuration section")
    logger.info(f"⏱️  Timeout set to {TIMEOUT_SECONDS} seconds per price batch")
    logger.info("🔄 Experiment will resume from where it left off if interrupted")
    logger.info("")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_entry()
