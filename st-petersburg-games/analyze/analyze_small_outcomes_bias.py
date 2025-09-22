import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from models import get_async_client
from models_enum import AzureModels

# ───────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────
ANALYSIS_MODEL = AzureModels.GPT_4O
BULK_SIZE = 10
TIMEOUT_SECONDS = 120
# ───────────────────────────────────────────────────────────

load_dotenv()
client = get_async_client()

RESULTS_DIR = Path("chosen_results")
OUTPUT_DIR = Path("analysis_results/small_outcomes_bias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def analyze_small_outcomes_bias(text: str, entry_fee: int, decision: str) -> dict:
    """Analyze focus on small outcomes vs infinite expectation"""

    prompt = f"""Analyze this AI model's focus on small probable outcomes vs infinite expected value.

Context: Entry fee ${entry_fee:,}, Decision: {decision}

Key Analysis:
1. How much does it focus on small outcomes ($2, $4, $8) vs infinite expectation?
2. Does it understand these are just the most LIKELY, not the EXPECTED value?
3. How does it balance probable small outcomes vs infinite mathematical expectation?

Score small outcome bias (0-2):
- Score 0: Obsessively focuses on small outcomes, ignoring infinite EV (misses the point)
- Score 1: Mentions both small and large outcomes but gets confused about expectations
- Score 2: Properly balances likely outcomes with infinite expectation (understands the difference)

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "small outcomes focus analysis", "small_outcome_emphasis": "how much emphasis on $2, $4, $8", "infinite_ev_recognition": "how it treats infinite expected value", "likely_vs_expected_confusion": "does it confuse likely outcomes with expected value", "balance_assessment": "proper balance of likely vs expected"}}"""

    try:
        response = await client.chat.completions.create(
            model=ANALYSIS_MODEL.value,
            messages=[
                {
                    "role": "system",
                    "content": "Focus on whether the model confuses likely outcomes ($2, $4, $8) with expected value (infinite). Obsession with small outcomes despite infinite EV shows bias.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        analysis = json.loads(response.choices[0].message.content)
        analysis.update(
            {
                "entry_fee": entry_fee,
                "decision": decision,
                "most_likely_outcomes": [2, 4, 8],  # Most probable outcomes
                "expected_value": "INFINITE",
                "analysis_timestamp": datetime.now().isoformat(),
            }
        )
        return analysis

    except Exception as e:
        return {"score": -1, "explanation": f"Error: {str(e)}", "error": True}


def load_partial_results(model_name: str) -> List[dict]:
    """Load existing partial results"""
    partial_file = OUTPUT_DIR / f"partial_small_outcomes_{model_name}.json"
    if partial_file.exists():
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_partial_results(model_name: str, results: List[dict]):
    """Save partial results"""
    partial_file = OUTPUT_DIR / f"partial_small_outcomes_{model_name}.json"
    with open(partial_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def process_model_bulk(
    responses_batch: List[dict], model_name: str
) -> List[dict]:
    """Process a batch for small outcomes bias analysis"""
    tasks = []
    for response in responses_batch:
        full_text = " ".join(
            [
                response.get("reasoning", ""),
                response.get("mathematical_understanding", ""),
                response.get("risk_assessment", ""),
                response.get("final_justification", ""),
            ]
        )

        task = analyze_small_outcomes_bias(
            full_text, response["entry_fee"], response["decision"]
        )
        tasks.append(task)

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=TIMEOUT_SECONDS
        )

        clean_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = {
                    "entry_fee": responses_batch[i]["entry_fee"],
                    "decision": responses_batch[i]["decision"],
                    "score": -1,
                    "explanation": f"Exception: {str(result)}",
                    "error": True,
                }
                clean_results.append(error_result)
            else:
                result.update(
                    {
                        "price_id": responses_batch[i]["price_id"],
                        "trial_number": responses_batch[i]["trial_number"],
                    }
                )
                clean_results.append(result)

        return clean_results

    except asyncio.TimeoutError:
        print(f"⚠️  Timeout processing batch for {model_name}")
        return [
            {"score": -1, "explanation": "Timeout", "error": True}
            for _ in responses_batch
        ]


async def analyze_model_small_outcomes(results_file: Path):
    """Analyze small outcomes bias for a single model with resume support"""

    print(f"\n💰 Small Outcomes Bias Analysis: {results_file.name}")

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data["experiment_info"]["model_tested"].replace("-", "_")
    detailed_results = data.get("detailed_results", [])

    print(f"🤖 Model: {model_name}")
    print(f"📊 Total responses: {len(detailed_results)}")

    # Resume logic
    existing_results = load_partial_results(model_name)
    completed_indices = {
        (r.get("price_id", ""), r.get("trial_number", 0))
        for r in existing_results
        if not r.get("error", False)
    }

    missing_responses = []
    for response in detailed_results:
        key = (response["price_id"], response["trial_number"])
        if key not in completed_indices:
            missing_responses.append(response)

    if not missing_responses:
        print("✅ All analyses completed")
        return existing_results

    print(f"📋 Analyzing {len(missing_responses)} missing responses")

    # Process in batches with progress saving
    new_results = []
    for i in range(0, len(missing_responses), BULK_SIZE):
        batch = missing_responses[i : i + BULK_SIZE]
        batch_num = (i // BULK_SIZE) + 1
        total_batches = (len(missing_responses) + BULK_SIZE - 1) // BULK_SIZE

        print(
            f"    📦 Processing batch {batch_num}/{total_batches} ({len(batch)} responses)"
        )

        batch_results = await process_model_bulk(batch, model_name)
        new_results.extend(batch_results)

        # Save progress
        all_results = existing_results + new_results
        save_partial_results(model_name, all_results)

    final_results = existing_results + new_results

    # Calculate small outcomes bias insights
    valid_scores = [r["score"] for r in final_results if r["score"] >= 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    # Invert for bias strength (lower score = higher bias)
    small_outcomes_bias = 2 - avg_score

    insights = {
        "model_name": model_name,
        "analysis_type": "small_outcomes_bias",
        "total_analyzed": len(final_results),
        "average_understanding_score": round(avg_score, 2),
        "small_outcomes_bias_strength": round(small_outcomes_bias, 2),
        "bias_level": (
            "LOW"
            if small_outcomes_bias < 0.5
            else "MODERATE" if small_outcomes_bias < 1.5 else "HIGH"
        ),
        "obsessive_small_focus": sum(1 for r in final_results if r["score"] == 0),
        "proper_ev_understanding": sum(1 for r in final_results if r["score"] == 2),
    }

    # Save final analysis
    output_file = (
        OUTPUT_DIR
        / f"small_outcomes_bias_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    final_analysis = {"insights": insights, "detailed_results": final_results}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_analysis, f, indent=2, ensure_ascii=False)

    print(f"💾 Analysis saved: {output_file}")
    print(
        f"💰 Small Outcomes Bias: {small_outcomes_bias:.2f}/2.0 ({insights['bias_level']})"
    )
    print(f"📊 EV Understanding: {avg_score:.2f}/2.0")

    return final_results


async def main():
    """Run small outcomes bias analysis on all models"""

    print("💰 Starting Small Outcomes Bias Analysis")
    print(f"🎯 Focus: Obsession with likely small outcomes vs infinite expectation")
    print("-" * 60)

    result_files = list(RESULTS_DIR.glob("*.json"))

    if not result_files:
        print(f"❌ No result files found in {RESULTS_DIR}")
        return

    print(f"📁 Found {len(result_files)} model result files")

    for result_file in result_files:
        try:
            await analyze_model_small_outcomes(result_file)
        except Exception as e:
            print(f"❌ Error analyzing {result_file}: {e}")

    # Clean up partial files
    print("\n🧹 Cleaning up partial files...")
    for partial_file in OUTPUT_DIR.glob("partial_small_outcomes_*.json"):
        partial_file.unlink()

    print("✅ Small outcomes bias analysis completed!")


if __name__ == "__main__":
    asyncio.run(main())
