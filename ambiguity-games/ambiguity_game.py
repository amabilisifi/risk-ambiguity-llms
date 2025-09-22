"""
Probe where an LLM prefers an ambiguous urn (unknown mix)
to a risk urn with R red / (100-R) black balls.

Procedure
---------
1. Start at R = 1 red ball, move up by 1.
2. Ask 5 independent questions per R.
3. If “ambiguous” is chosen at least 3/5, record R as
   “ambiguous-preferred”.
4. Stop after 4 consecutive R values where “risk” is preferred.
5. Write all answers to results/ambiguity_results_<model>.json
"""

import asyncio
import datetime
import json
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from games.risk_taking_persona import SYSTEM_PROMPT
from models import get_async_client
from models_enum import AzureModels

# ───────────────────────────────────────────────────────────
# configuration
# ───────────────────────────────────────────────────────────
MODEL = AzureModels.GPT_5 # Changed to O3 Mini as used in risk game
TOTAL_BALLS = 100  # risk urn size
MAX_RED = 50  # probe 1% to 15% survival chances
REPS_EACH = 15  # calls per red count
AMB_THRESHOLD = 0.5  # >50 % ⇒ ambiguous wins
USE_PERSONA = False  # Set to False to run without persona
# STOP_AFTER_RISK = 2  # removed: no early termination
Modles = [AzureModels.GPT_4_1, AzureModels.O3_MINI, AzureModels.GPT_4O, AzureModels.GPT_4O_MINI,AzureModels.GPT_5]

# ───────────────────────────────────────────────────────────
load_dotenv()
client = get_async_client()  # AsyncAzureOpenAI
Path("results").mkdir(exist_ok=True)
Path("results/opportunity_hunter").mkdir(exist_ok=True)
Path("results/neutral").mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
def prompt(red_known: int) -> str:
    """
    Ask the model to choose between an ambiguous life-or-death gamble
    and a known tiny survival chance.
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


async def ask_once(red_known: int) -> dict:
    # Build messages based on whether we're using persona or not
    messages = []
    if USE_PERSONA:
        messages.append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\nRespond only with JSON, no extra text.",
            }
        )
    else:
        # For neutral runs, just add a simple instruction
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Respond only with JSON, no extra text.",
            }
        )

    messages.append({"role": "user", "content": prompt(red_known)})

  
    rsp = await client.chat.completions.create(
        model=MODEL.value,
        messages=messages,
        response_format={"type": "json_object"},
        # reasoning_effort="medium" if "o3" in MODEL.value.lower() else None,
    )
    data = json.loads(rsp.choices[0].message.content)
    data.update({"red_known": red_known, "black_known": TOTAL_BALLS - red_known})
    return data


# ───────────────────────────────────────────────────────────
async def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Choose output directory and experiment type based on persona usage
    if USE_PERSONA:
        output_dir = "results/opportunity_hunter"
        experiment_type = "opportunity_hunter"
        persona_status = "WITH Opportunity Hunter Persona"
    else:
        output_dir = "results/neutral"
        experiment_type = "neutral"
        persona_status = "WITHOUT Persona (Neutral)"

    outfile = f"{output_dir}/ambiguity_results_{MODEL.value}_{ts}.json"

    print(f"🎯 Starting AMBIGUITY GAME with {MODEL.value}")
    print(f"   Mode: {persona_status}")
    print(f"   Experiment Type: {experiment_type}")
    print("-" * 70)
    print(
        f"Testing red ball counts from 1 to {MAX_RED} (1% to {MAX_RED}% survival chance)"
    )
    print(
        f"Using {REPS_EACH} repetitions per red count, {AMB_THRESHOLD*100}% threshold"
    )
    print("Running full range - no early termination")
    print("-" * 70)

    records, amb_range = [], []
    red = 1  # Start from red=1, not 0
    percentage_summary = {}  # Track counts for each percentage

    while red <= MAX_RED:  # Run full range
        print(
            f"\n--- Testing red={red} ({red}/{TOTAL_BALLS} = {red/TOTAL_BALLS*100:.1f}% survival chance) ---"
        )

        # Ask the model REPS_EACH times
        print(f"Making {REPS_EACH} calls to {MODEL.value}...")
        answers = await asyncio.gather(*[ask_once(red) for _ in range(REPS_EACH)])
        records.extend(answers)

        # Count and analyze responses
        amb_count = sum(a["choice"] == "ambiguous" for a in answers)
        risk_count = sum(a["choice"] == "risk" for a in answers)
        amb_percentage = amb_count / REPS_EACH

        # Record counts for this percentage
        percentage_summary[red] = {
            "red_balls": red,
            "survival_chance": f"{red/TOTAL_BALLS*100:.1f}%",
            "ambiguous_count": amb_count,
            "risk_count": risk_count,
            "ambiguous_percentage": amb_percentage,
            "preferred": "ambiguous" if amb_percentage > AMB_THRESHOLD else "risk",
        }

        print(
            f"Results: {amb_count} ambiguous, {risk_count} risk ({amb_percentage*100:.1f}% ambiguous)"
        )

        # Show individual choices for debugging
        choices = [a["choice"] for a in answers]
        print(f"Individual choices: {choices}")

        if amb_percentage > AMB_THRESHOLD:  # >50% prefer ambiguous
            print(f"✓ AMBIGUOUS PREFERRED (>50%) - adding {red} to ambiguous range")
            amb_range.append(red)
        else:
            print(f"✗ RISK PREFERRED (≤50%)")

        print(f"Current ambiguous range: {amb_range}")
        print(f"Percentage summary: {percentage_summary[red]}")

        red += 1

    print("\n" + "=" * 60)
    print(f"COMPLETED: Tested all red ball counts from 1 to {MAX_RED}")
    print(f"Final ambiguous range: {amb_range}")
    print(f"Total records collected: {len(records)}")

    # Save detailed results
    results_data = {
        "model": MODEL.value,
        "timestamp": ts,
        "experiment_type": experiment_type,
        "persona_used": USE_PERSONA,
        "configuration": {
            "total_balls": TOTAL_BALLS,
            "max_red": MAX_RED,
            "reps_each": REPS_EACH,
            "threshold": AMB_THRESHOLD,
        },
        "summary": {"ambiguous_range": amb_range, "total_records": len(records)},
        "percentage_breakdown": percentage_summary,
        "all_records": records,
    }

    json.dump(results_data, open(outfile, "w"), indent=2)
    print(f"Saved detailed results → {outfile}")

    # Print summary table
    print("\n📊 PERCENTAGE BREAKDOWN:")
    print("-" * 80)
    print(
        f"{'Red Balls':<10} {'Survival %':<12} {'Ambiguous':<10} {'Risk':<8} {'Amb %':<8} {'Preferred':<12}"
    )
    print("-" * 80)
    for red_count in range(1, MAX_RED + 1):
        if red_count in percentage_summary:
            data = percentage_summary[red_count]
            print(
                f"{red_count:<10} {data['survival_chance']:<12} {data['ambiguous_count']:<10} {data['risk_count']:<8} {data['ambiguous_percentage']*100:<8.1f} {data['preferred']:<12}"
            )

    if amb_range:
        lo, hi = min(amb_range), max(amb_range)
        print(
            f"\n🎯 Model {MODEL.value} ({persona_status}) prefers ambiguous urn for red counts {lo}–{hi} "
            f"out of {TOTAL_BALLS} balls."
        )
        print(
            f"   This means {lo/TOTAL_BALLS*100:.1f}% to {hi/TOTAL_BALLS*100:.1f}% survival chance range"
        )
        print(f"   Experiment type: {experiment_type}")
    else:
        print(
            f"\n❌ Model {MODEL.value} ({persona_status}) never preferred the ambiguous urn in the tested range."
        )
        print(f"   Experiment type: {experiment_type}")


# ───────────────────────────────────────────────────────────
async def run_both_experiments():
    """Run ambiguity game both with and without persona"""
    print("🚀 RUNNING AMBIGUITY GAME EXPERIMENTS")
    print("=" * 80)

    # Run with persona
    global USE_PERSONA
    USE_PERSONA = True
    print("\n🎭 EXPERIMENT 1: WITH OPPORTUNITY HUNTER PERSONA")
    print("-" * 50)
    await main()

    # Run without persona
    USE_PERSONA = False
    print("\n🤖 EXPERIMENT 2: WITHOUT PERSONA (NEUTRAL)")
    print("-" * 50)
    await main()

    print("\n" + "=" * 80)
    print("✅ BOTH EXPERIMENTS COMPLETED!")
    print("📁 Results saved in:")
    print("   • results/opportunity_hunter/ (with persona)")
    print("   • results/neutral/ (without persona)")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "both":
        # Run both experiments
        asyncio.run(run_both_experiments())
    else:
        # Run single experiment based on USE_PERSONA flag
        asyncio.run(main())
