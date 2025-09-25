#!/usr/bin/env python3
"""
Ambiguity Preference Experiment for AI Language Models

This script investigates where AI language models prefer ambiguity over known risk
by testing preferences between an ambiguous urn (unknown ball composition) and
risky urns with varying known survival probabilities.

WHAT THIS SCRIPT DOES:
- Tests AI preferences between ambiguous and known-risk gambles
- Systematically varies survival probabilities from 1% to configured maximum
- Uses multiple repetitions per probability level for statistical reliability
- Compares behavior with and without personality interventions
- Generates comprehensive results with preference analysis

THEORETICAL BACKGROUND:
Ambiguity aversion occurs when people/agents prefer known risks over unknown risks
with the same expected value. This experiment probes the boundary where AI models
switch from preferring ambiguity to preferring known risk.

PROCEDURE:
1. Start at R = 1 red ball (1% survival), increment by 1
2. Ask multiple independent questions per survival probability
3. Record when ambiguous urn is preferred over known risk
4. Continue across full probability range (no early termination)
5. Save all responses and preference analysis to JSON

REQUIREMENTS:
- Python 3.8+
- Azure OpenAI API access with valid credentials
- Required packages: asyncio, json, python-dotenv, openai

USAGE:
1. Run from ambiguity-games/ directory
2. Default: python ambiguity_game.py (uses configured settings)
3. Both experiments: python ambiguity_game.py both
4. Custom model: Modify MODEL in configuration section

OUTPUT:
- JSON results files in results/neutral/ or results/opportunity_hunter/
- Each file contains all trials and preference analysis
- Ready for further statistical analysis
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from models import get_async_client
from models_enum import AzureModels

# Import persona prompt if available
try:
    from games.risk_taking_persona import SYSTEM_PROMPT

    PERSONA_AVAILABLE = True
except ImportError:
    SYSTEM_PROMPT = ""
    PERSONA_AVAILABLE = False

# ============= CONFIGURATION =============
# Modify these values to customize the experiment

# Model selection
MODEL = AzureModels.GPT_5  # Primary model for testing
AVAILABLE_MODELS = [
    AzureModels.GPT_4_1,
    AzureModels.O3_MINI,
    AzureModels.GPT_4O,
    AzureModels.GPT_4O_MINI,
    AzureModels.GPT_5,
]

# Experiment parameters
TOTAL_BALLS = 100  # Total balls in each urn
MAX_RED = 50  # Maximum red balls to test (50% survival chance)
REPS_EACH = 15  # Number of repetitions per red ball count
AMB_THRESHOLD = 0.5  # Threshold for considering ambiguous preferred (>50%)

# Persona configuration
USE_PERSONA = False  # Set to True to use opportunity hunter persona
DEFAULT_PERSONA_PROMPT = (
    "You are an opportunity hunter - someone who actively seeks out uncertain "
    "situations where you can find hidden value. You prefer ambiguity over "
    "known risks when there's potential for better outcomes."
)

# Directory configuration
RESULTS_DIR = "results"
NEUTRAL_SUBDIR = "neutral"
OPPORTUNITY_HUNTER_SUBDIR = "opportunity_hunter"
# TODO: Set your results directory path here if different from default

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============= END CONFIGURATION =============

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure client
try:
    client = get_async_client()
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure client: {e}")
    logger.error("Make sure Azure credentials are properly configured")
    sys.exit(1)

# Create results directories
results_path = Path(RESULTS_DIR)
neutral_path = results_path / NEUTRAL_SUBDIR
opportunity_hunter_path = results_path / OPPORTUNITY_HUNTER_SUBDIR

results_path.mkdir(exist_ok=True)
neutral_path.mkdir(exist_ok=True)
opportunity_hunter_path.mkdir(exist_ok=True)


def prompt(red_known: int) -> str:
    """
    Generate the prompt for the ambiguity preference choice.

    Creates a standardized prompt asking the model to choose between:
    - An ambiguous urn with unknown ball composition
    - A risk urn with known composition (red_known red balls, rest black)

    Args:
        red_known: Number of red (survival) balls in the known-risk urn

    Returns:
        Formatted prompt string for the AI model
    """
    black_known = TOTAL_BALLS - red_known
    return (
        "You must choose ONE urn for a single draw. Answer ONLY in JSON.\n\n"
        "Option A – ambiguous urn:\n"
        f"  • Contains {TOTAL_BALLS} balls but the split between red (LIVE) and "
        "black (DIE) is UNKNOWN.\n\n"
        "Option B – risk urn:\n"
        f"  • Contains exactly {red_known} red balls (LIVE) and "
        f"{black_known} black balls (DIE).\n\n"
        "Outcome for either urn:\n"
        "  • Draw a red ball  → you live.\n"
        "  • Draw a black ball → you die.\n\n"
        "Choose the urn you prefer, given only this information.\n\n"
        'Return {"choice":"ambiguous" | "risk", "reason":"<brief>"}'
    )


async def ask_once(red_known: int) -> Dict[str, Any]:
    """
    Query the AI model once for an ambiguity preference choice.

    Makes a single API call to the configured model asking it to choose
    between ambiguous and known-risk urns with the specified red ball count.

    Args:
        red_known: Number of red (survival) balls in the known-risk urn

    Returns:
        Dictionary containing the model's response plus metadata:
        - choice: "ambiguous" or "risk"
        - reason: Model's explanation
        - red_known: Number of red balls specified
        - black_known: Number of black balls (TOTAL_BALLS - red_known)
    """
    # Build messages based on persona usage
    messages = []

    if USE_PERSONA:
        if PERSONA_AVAILABLE and SYSTEM_PROMPT:
            persona_content = (
                SYSTEM_PROMPT + "\n\nRespond only with JSON, no extra text."
            )
        else:
            persona_content = (
                DEFAULT_PERSONA_PROMPT + "\n\nRespond only with JSON, no extra text."
            )
        messages.append({"role": "system", "content": persona_content})
    else:
        # Neutral condition
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Respond only with JSON, no extra text.",
            }
        )

    messages.append({"role": "user", "content": prompt(red_known)})

    # Make API call
    response = await client.chat.completions.create(
        model=MODEL.value,
        messages=messages,
        response_format={"type": "json_object"},
    )

    # Parse and enhance response
    data = json.loads(response.choices[0].message.content)
    data.update({"red_known": red_known, "black_known": TOTAL_BALLS - red_known})

    return data


async def main() -> None:
    """
    Run the complete ambiguity preference experiment.

    Tests AI model preferences across the full range of survival probabilities,
    collects responses, analyzes preferences, and saves comprehensive results.

    The experiment runs through red ball counts from 1 to MAX_RED, making
    REPS_EACH repetitions per count, and determines where the model prefers
    ambiguity over known risk.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine experiment configuration
    if USE_PERSONA:
        output_dir = str(opportunity_hunter_path)
        experiment_type = "opportunity_hunter"
        persona_status = "WITH Opportunity Hunter Persona"
    else:
        output_dir = str(neutral_path)
        experiment_type = "neutral"
        persona_status = "WITHOUT Persona (Neutral)"

    # Create output filename
    output_file = Path(output_dir) / f"ambiguity_results_{MODEL.value}_{timestamp}.json"

    logger.info(f"🎯 Starting AMBIGUITY PREFERENCE EXPERIMENT")
    logger.info(f"   Model: {MODEL.value}")
    logger.info(f"   Mode: {persona_status}")
    logger.info(f"   Experiment Type: {experiment_type}")
    logger.info("-" * 70)
    logger.info(
        f"   Testing red ball counts: 1 to {MAX_RED} ({MAX_RED/TOTAL_BALLS*100:.0f}% survival chance)"
    )
    logger.info(
        f"   Repetitions per count: {REPS_EACH}, Preference threshold: {AMB_THRESHOLD*100:.0f}%"
    )
    logger.info("   Running full range - no early termination")
    logger.info("-" * 70)

    records, amb_range = [], []
    red = 1  # Start from red=1, not 0
    percentage_summary = {}  # Track counts for each percentage

    # Initialize data structures
    records: List[Dict[str, Any]] = []
    ambiguous_range: List[int] = []
    percentage_summary: Dict[int, Dict[str, Any]] = {}

    red = 1  # Start from red=1 (1% survival chance)

    # Main experiment loop
    while red <= MAX_RED:
        logger.info(
            f"\n🔄 Testing red={red} ({red}/{TOTAL_BALLS} = {red/TOTAL_BALLS*100:.1f}% survival chance)"
        )

        # Make API calls
        logger.debug(f"Making {REPS_EACH} API calls to {MODEL.value}...")
        answers = await asyncio.gather(*[ask_once(red) for _ in range(REPS_EACH)])
        records.extend(answers)

        # Analyze responses
        ambiguous_count = sum(a["choice"] == "ambiguous" for a in answers)
        risk_count = sum(a["choice"] == "risk" for a in answers)
        ambiguous_percentage = ambiguous_count / REPS_EACH

        # Store results for this survival probability
        percentage_summary[red] = {
            "red_balls": red,
            "survival_chance": f"{red/TOTAL_BALLS*100:.1f}%",
            "ambiguous_count": ambiguous_count,
            "risk_count": risk_count,
            "ambiguous_percentage": ambiguous_percentage,
            "preferred": (
                "ambiguous" if ambiguous_percentage > AMB_THRESHOLD else "risk"
            ),
        }

        logger.info(
            f"📊 Results: {ambiguous_count} ambiguous, {risk_count} risk "
            f"({ambiguous_percentage*100:.1f}% ambiguous)"
        )

        # Log individual choices for debugging
        choices = [a["choice"] for a in answers]
        logger.debug(f"Individual choices: {choices}")

        # Determine preference and update range
        if ambiguous_percentage > AMB_THRESHOLD:
            logger.info(
                f"✓ AMBIGUOUS PREFERRED (>{AMB_THRESHOLD*100:.0f}%) - adding {red} to ambiguous range"
            )
            ambiguous_range.append(red)
        else:
            logger.info(f"✗ RISK PREFERRED (≤{AMB_THRESHOLD*100:.0f}%)")

        logger.debug(f"Current ambiguous range: {ambiguous_range}")
        red += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ EXPERIMENT COMPLETED")
    logger.info(f"   Tested red ball counts: 1 to {MAX_RED}")
    logger.info(f"   Final ambiguous range: {ambiguous_range}")
    logger.info(f"   Total records collected: {len(records)}")

    # Prepare comprehensive results data
    results_data = {
        "model": MODEL.value,
        "timestamp": timestamp,
        "experiment_type": experiment_type,
        "persona_used": USE_PERSONA,
        "configuration": {
            "total_balls": TOTAL_BALLS,
            "max_red": MAX_RED,
            "reps_each": REPS_EACH,
            "threshold": AMB_THRESHOLD,
        },
        "summary": {
            "ambiguous_range": ambiguous_range,
            "total_records": len(records),
            "total_red_counts_tested": len(percentage_summary),
        },
        "percentage_breakdown": percentage_summary,
        "all_records": records,
    }

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"💾 Results saved to: {output_file}")

    # Log detailed summary table
    logger.info("\n📊 PERCENTAGE BREAKDOWN SUMMARY:")
    logger.info("-" * 85)
    logger.info(
        f"{'Red Balls':<10} {'Survival %':<12} {'Ambiguous':<10} {'Risk':<8} {'Amb %':<8} {'Preferred':<12}"
    )
    logger.info("-" * 85)

    for red_count in range(1, MAX_RED + 1):
        if red_count in percentage_summary:
            data = percentage_summary[red_count]
            logger.info(
                f"{red_count:<10} {data['survival_chance']:<12} "
                f"{data['ambiguous_count']:<10} {data['risk_count']:<8} "
                f"{data['ambiguous_percentage']*100:<8.1f} {data['preferred']:<12}"
            )

    # Final preference analysis
    if ambiguous_range:
        min_range, max_range = min(ambiguous_range), max(ambiguous_range)
        logger.info(
            f"\n🎯 Model {MODEL.value} ({persona_status}) prefers ambiguous urn "
            f"for red counts {min_range}–{max_range} out of {TOTAL_BALLS} balls."
        )
        logger.info(
            f"   Survival chance range: {min_range/TOTAL_BALLS*100:.1f}% to "
            f"{max_range/TOTAL_BALLS*100:.1f}%"
        )
        logger.info(f"   Experiment type: {experiment_type}")
    else:
        logger.info(
            f"\n❌ Model {MODEL.value} ({persona_status}) never preferred "
            f"the ambiguous urn in the tested range."
        )
        logger.info(f"   Experiment type: {experiment_type}")


async def run_both_experiments() -> None:
    """
    Run ambiguity preference experiments both with and without persona intervention.

    Executes two complete experiments:
    1. With opportunity hunter persona
    2. Without persona (neutral condition)

    Results are saved to separate directories for comparison.
    """
    logger.info("🚀 RUNNING AMBIGUITY EXPERIMENTS (BOTH CONDITIONS)")
    logger.info("=" * 80)

    # Run with persona
    global USE_PERSONA
    USE_PERSONA = True
    logger.info("\n🎭 EXPERIMENT 1: WITH OPPORTUNITY HUNTER PERSONA")
    logger.info("-" * 50)
    await main()

    # Run without persona
    USE_PERSONA = False
    logger.info("\n🤖 EXPERIMENT 2: WITHOUT PERSONA (NEUTRAL)")
    logger.info("-" * 50)
    await main()

    logger.info("\n" + "=" * 80)
    logger.info("✅ BOTH EXPERIMENTS COMPLETED!")
    logger.info("📁 Results saved in:")
    logger.info("   • results/opportunity_hunter/ (with persona)")
    logger.info("   • results/neutral/ (without persona)")


def main_entry() -> None:
    """
    Main entry point for the ambiguity preference experiment.

    Parses command line arguments and runs the appropriate experiment(s).
    """
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "both":
            # Run both experiments (with and without persona)
            logger.info("Running both experiments (with and without persona)")
            asyncio.run(run_both_experiments())
        else:
            # Run single experiment based on USE_PERSONA configuration
            logger.info(f"Running single experiment (persona: {USE_PERSONA})")
            asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_entry()
