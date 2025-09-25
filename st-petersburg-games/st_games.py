import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from risk_taking_persona import SYSTEM_PROMPT

from models import get_async_client
from models_enum import AzureModels

SELECTED_MODELS = [
    # AzureModels.O3_MINI.value,  # o3-mini with medium reasoning
    AzureModels.GPT_5.value,  # gpt-5
    AzureModels.GPT_4O_MINI.value,  # 4o-mini
    AzureModels.GPT_4O.value,  # 4o
    AzureModels.GPT_4_1.value,  # 4.1
]

N_TRIALS_PER_PRICE = 10
TIMEOUT_SECONDS = 180

load_dotenv()
client = get_async_client()

for model in SELECTED_MODELS:
    safe_model_name = model.replace(".", "-").replace("/", "-")
    model_dir = Path("results") / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

# -------- St. Petersburg Paradox Entry Prices --------
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
    price_case: dict, trial_number: int, model_name: str, max_retries: int = 3
) -> dict:
    """Send async request to model for St. Petersburg decision with retry logic"""

    for attempt in range(max_retries):
        try:
            # Prepare API parameters
            api_params = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "system",
                        "content": "You are participating in a decision-making experiment about the St. Petersburg Paradox. Respond only with valid JSON containing your decision: either PLAY or PASS.",
                    },
                    {
                        "role": "user",
                        "content": create_st_petersburg_prompt(price_case["entry_fee"]),
                    },
                ],
                "response_format": {"type": "json_object"},
            }

            # Make the API call with timeout
            api_call = client.chat.completions.create(**api_params)
            response = await asyncio.wait_for(api_call, timeout=TIMEOUT_SECONDS)

            model_response = json.loads(response.choices[0].message.content)

            # Extract decision from the simplified format
            decision = model_response.get("decision", "ERROR").upper()

            # Ensure decision is valid
            if decision not in ["PLAY", "PASS"]:
                decision = "ERROR"

            # Add metadata to the response
            result = {
                "price_id": price_case["id"],
                "entry_fee": price_case["entry_fee"],
                "trial_number": trial_number,
                "timestamp": datetime.now().isoformat(),
                "attempt": attempt + 1,  # Track which attempt succeeded
                "Decision": decision,
                "Entrance Fee": price_case["entry_fee"],
            }

            return result

        except asyncio.TimeoutError:
            print(
                f"⏰ Timeout on trial {trial_number} for ${price_case['entry_fee']:,} ({model_name}, attempt {attempt + 1}/{max_retries})"
            )
            if attempt == max_retries - 1:  # Last attempt failed
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
            # Wait a bit before retry
            await asyncio.sleep(1)

        except Exception as e:
            print(
                f"❌ Error in trial {trial_number} for ${price_case['entry_fee']:,} ({model_name}, attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:  # Last attempt failed
                return {
                    "price_id": price_case["id"],
                    "entry_fee": price_case["entry_fee"],
                    "trial_number": trial_number,
                    "timestamp": datetime.now().isoformat(),
                    "Entrance Fee": price_case["entry_fee"],
                    "Decision": "ERROR",
                    "error": True,
                }
            # Wait a bit before retry
            await asyncio.sleep(1)

    # This should never be reached, but just in case
    return {
        "price_id": price_case["id"],
        "entry_fee": price_case["entry_fee"],
        "trial_number": trial_number,
        "timestamp": datetime.now().isoformat(),
        "Entrance Fee": price_case["entry_fee"],
        "Decision": "ERROR",
        "error": True,
    }


def load_partial_results(price_case: dict, model_name: str) -> List[dict]:
    # Clean model name for filename
    safe_model_name = model_name.replace(".", "-").replace("/", "-")
    model_dir = Path("results") / safe_model_name
    partial_file = model_dir / f"partial_{price_case['id']}.json"

    if partial_file.exists():
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []


def save_partial_results(price_case: dict, results: List[dict], model_name: str):
    safe_model_name = model_name.replace(".", "-").replace("/", "-")
    model_dir = Path("results") / safe_model_name
    partial_file = model_dir / f"partial_{price_case['id']}.json"

    with open(partial_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def run_price_trials_with_timeout(
    price_case: dict, model_name: str
) -> List[dict]:

    print(
        f"Starting trials for entry price: ${price_case['entry_fee']:,} ({model_name})"
    )

    # Load any existing results
    existing_results = load_partial_results(price_case, model_name)
    completed_trials = {r["trial_number"] for r in existing_results}

    if len(completed_trials) == N_TRIALS_PER_PRICE:
        print(
            f"✅ All {N_TRIALS_PER_PRICE} trials already completed for ${price_case['entry_fee']:,}"
        )
        return existing_results

    # Find missing trials
    missing_trials = [
        t for t in range(1, N_TRIALS_PER_PRICE + 1) if t not in completed_trials
    ]

    if completed_trials:
        print(
            f"📋 Found {len(completed_trials)} existing results, running {len(missing_trials)} missing trials: {missing_trials}"
        )
    else:
        print(f"🆕 Running all {N_TRIALS_PER_PRICE} trials")

    # Create tasks for missing trials only
    print(
        f"⏱️  Starting {len(missing_trials)} requests with {TIMEOUT_SECONDS}s timeout..."
    )

    successful_results = []
    retry_trials = missing_trials.copy()
    max_retries = 3

    for retry_attempt in range(max_retries):
        if not retry_trials:
            break

        print(
            f"🔄 Attempt {retry_attempt + 1}/{max_retries}: Processing {len(retry_trials)} trials"
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
                    print(f"   ❌ Trial {trial_num} failed: {str(result)}")
                    failed_trials.append(trial_num)
                else:
                    print(f"   ✅ Trial {trial_num} completed")
                    successful_results.append(result)
                    completed_trials.append(trial_num)

            # Update retry list to only failed trials
            retry_trials = failed_trials

            print(
                f"   📊 Batch {retry_attempt + 1}: {len(completed_trials)} completed, {len(failed_trials)} failed"
            )

            # Save progress after each batch
            all_results = existing_results + successful_results
            save_partial_results(price_case, all_results, model_name)

        except Exception as e:
            print(f"   ❌ Batch {retry_attempt + 1} failed completely: {e}")
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

    print(
        f"✅ Completed {len(successful_results)} trials for ${price_case['entry_fee']:,} ({len([r for r in successful_results if r.get('Decision') != 'ERROR'])} successful)"
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


async def main():
    """Run the complete St. Petersburg Paradox experiment with separate results per model"""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🎯 Starting Multi-Model St. Petersburg Paradox Experiment")
    print(f"🤖 Models: {len(SELECTED_MODELS)} AI systems")
    print(f"📊 Testing {len(ENTRY_PRICES)} different entry prices per model")
    print(f"🔄 Running {N_TRIALS_PER_PRICE} trials per price per model")
    print(f"⏱️  Timeout: {TIMEOUT_SECONDS} seconds per price")
    total_trials = len(SELECTED_MODELS) * len(ENTRY_PRICES) * N_TRIALS_PER_PRICE
    print(f"📈 Total trials across all models: {total_trials:,}")

    print("\n🤖 SELECTED MODELS:")
    for i, model in enumerate(SELECTED_MODELS, 1):
        print(f"   {i}. {model}")

    print("-" * 60)

    # Run experiment for each model separately
    for model_idx, model_name in enumerate(SELECTED_MODELS, 1):
        print(f"\n🎯 MODEL {model_idx}/{len(SELECTED_MODELS)}: {model_name}")

        try:
            # Run experiment for this model
            model_results = await run_experiment_for_model(model_name)

            # Save individual model results
            safe_model_name = model_name.replace(".", "-").replace("/", "-")
            model_dir = Path("results") / safe_model_name
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

            print(f"✅ Completed {model_name}: {len(model_results)} total trials")
            print(f"📁 Results saved to: {outfile}")

        except Exception as e:
            print(f"❌ Failed to process {model_name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("🎉 MULTI-MODEL EXPERIMENT COMPLETED!")
    print(f"🤖 Models tested: {len(SELECTED_MODELS)}")
    print(f"📁 Each model's results saved to: results/{{model_name}}/")

    # Clean up partial files after successful completion
    print("🧹 Cleaning up partial result files...")
    for model_name in SELECTED_MODELS:
        safe_model_name = model_name.replace(".", "-").replace("/", "-")
        model_dir = Path("results") / safe_model_name
        for price_case in ENTRY_PRICES:
            partial_file = model_dir / f"partial_{price_case['id']}.json"
            if partial_file.exists():
                partial_file.unlink()

    print("✅ All partial files cleaned up!")
    print("=" * 80)


if __name__ == "__main__":
    print(f"🚀 Starting Multi-Model St. Petersburg Paradox Experiment")
    print(f"🤖 Testing {len(SELECTED_MODELS)} models: {', '.join(SELECTED_MODELS)}")
    print("💡 To change models, modify SELECTED_MODELS in configuration section")
    print(f"⏱️  Timeout set to {TIMEOUT_SECONDS} seconds per price batch")
    print("🔄 Experiment will resume from where it left off if interrupted")
    print()

    asyncio.run(main())
