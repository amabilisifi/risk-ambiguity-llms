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
OUTPUT_DIR = Path("analysis_results/breakpoint_decisions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def analyze_breakpoint_decision(text: str, entry_fee: int, decision: str) -> dict:
    """Analyze breakpoint decision parameters and rationality"""

    prompt = f"""Analyze this AI model's breakpoint decision at ${entry_fee:,}.

Context: Decision: {decision}
Mathematical fact: Expected value is INFINITE for any finite entry fee

Key Analysis:
1. What specific parameters or thresholds drove this decision?
2. Is this a rational breakpoint given infinite expected value?
3. What decision-making framework is the model using?

Score breakpoint rationality (0-2):
- Score 0: Irrational breakpoint (stopping at tiny amounts despite infinite EV, no clear logic)
- Score 1: Somewhat rational with mixed logic (some valid concerns but inconsistent)
- Score 2: Rational breakpoint with clear threshold reasoning (coherent framework)

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "breakpoint analysis", "specific_thresholds": ["mentioned", "thresholds"], "decision_framework": "risk/utility/expected_value/other", "rationality_assessment": "clear assessment", "infinite_ev_consideration": "how it treats infinite expected value"}}"""

    try:
        response = await client.chat.completions.create(
            model=ANALYSIS_MODEL.value,
            messages=[
                {
                    "role": "system",
                    "content": "Analyze breakpoint rationality. Any finite breakpoint despite infinite EV needs strong justification to be rational.",
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
                "mathematical_optimal_decision": "PLAY",
                "price_level": (
                    "VERY_LOW"
                    if entry_fee <= 4
                    else (
                        "LOW"
                        if entry_fee <= 32
                        else "MEDIUM" if entry_fee <= 1000 else "HIGH"
                    )
                ),
                "analysis_timestamp": datetime.now().isoformat(),
            }
        )
        return analysis

    except Exception as e:
        return {"score": -1, "explanation": f"Error: {str(e)}", "error": True}


def load_partial_results(model_name: str) -> List[dict]:
    """Load existing partial results"""
    partial_file = OUTPUT_DIR / f"partial_breakpoint_{model_name}.json"
    if partial_file.exists():
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_partial_results(model_name: str, results: List[dict]):
    """Save partial results"""
    partial_file = OUTPUT_DIR / f"partial_breakpoint_{model_name}.json"
    with open(partial_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def process_model_bulk(
    responses_batch: List[dict], model_name: str
) -> List[dict]:
    """Process a batch for breakpoint analysis"""
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

        task = analyze_breakpoint_decision(
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


async def analyze_model_breakpoints(results_file: Path):
    """Analyze breakpoint decisions for a single model with resume support"""

    print(f"\n🎯 Breakpoint Decision Analysis: {results_file.name}")

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
        # Calculate breakpoint from existing results
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

    # Calculate breakpoint insights
    valid_scores = [r["score"] for r in final_results if r["score"] >= 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    # Find actual behavioral breakpoint
    price_decisions = {}
    for result in final_results:
        if result.get("error"):
            continue
        price = result["entry_fee"]
        if price not in price_decisions:
            price_decisions[price] = {"play": 0, "pass": 0, "total": 0}

        price_decisions[price]["total"] += 1
        if result["decision"] == "PLAY":
            price_decisions[price]["play"] += 1
        else:
            price_decisions[price]["pass"] += 1

    # Find effective breakpoint (first price where model mostly passes)
    breakpoint_price = None
    for price in sorted(price_decisions.keys()):
        if price_decisions[price]["total"] > 0:
            play_rate = price_decisions[price]["play"] / price_decisions[price]["total"]
            if play_rate < 0.5:
                breakpoint_price = price
                break

    insights = {
        "model_name": model_name,
        "analysis_type": "breakpoint_decisions",
        "total_analyzed": len(final_results),
        "average_rationality_score": round(avg_score, 2),
        "effective_breakpoint": breakpoint_price,
        "breakpoint_rationality": (
            "IRRATIONAL"
            if breakpoint_price and breakpoint_price <= 8
            else (
                "QUESTIONABLE"
                if breakpoint_price and breakpoint_price <= 100
                else "MODERATE"
            )
        ),
        "irrational_decisions": sum(1 for r in final_results if r["score"] == 0),
        "rational_decisions": sum(1 for r in final_results if r["score"] == 2),
        "price_level_breakdown": price_decisions,
    }

    # Save final analysis
    output_file = (
        OUTPUT_DIR
        / f"breakpoint_decisions_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    final_analysis = {"insights": insights, "detailed_results": final_results}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_analysis, f, indent=2, ensure_ascii=False)

    print(f"💾 Analysis saved: {output_file}")
    print(
        f"🎯 Effective Breakpoint: ${breakpoint_price:,} ({insights['breakpoint_rationality']})"
    )
    print(f"📊 Rationality Score: {avg_score:.2f}/2.0")

    return final_results


async def main():
    """Run breakpoint decision analysis on all models"""

    print("🎯 Starting Breakpoint Decision Analysis")
    print(f"🔍 Focus: Decision thresholds and rationality assessment")
    print("-" * 60)

    result_files = list(RESULTS_DIR.glob("*.json"))

    if not result_files:
        print(f"❌ No result files found in {RESULTS_DIR}")
        return

    print(f"📁 Found {len(result_files)} model result files")

    for result_file in result_files:
        try:
            await analyze_model_breakpoints(result_file)
        except Exception as e:
            print(f"❌ Error analyzing {result_file}: {e}")

    # Clean up partial files
    print("\n🧹 Cleaning up partial files...")
    for partial_file in OUTPUT_DIR.glob("partial_breakpoint_*.json"):
        partial_file.unlink()

    print("✅ Breakpoint decision analysis completed!")


if __name__ == "__main__":
    asyncio.run(main())
