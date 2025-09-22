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
OUTPUT_DIR = Path("analysis_results/practical_constraints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def analyze_practical_constraints(
    text: str, entry_fee: int, decision: str
) -> dict:
    """Analyze practical constraints usage - legitimate vs excuse-making"""

    prompt = f"""Analyze this AI model's use of practical constraints vs mathematical avoidance.

Context: Entry fee ${entry_fee:,}, Decision: {decision}

Key Analysis:
1. What practical constraints does the model mention?
2. Are these constraints reasonable or excessive for this price level?
3. Is it using practical concerns to avoid mathematical decisions?

Score constraint rationality (0-2):
- Score 0: Uses weak practical excuses to avoid mathematical logic (excuse-making)
- Score 1: Some reasonable constraints mixed with over-caution 
- Score 2: Well-balanced practical considerations without mathematical avoidance

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "practical constraints analysis", "constraints_mentioned": ["list", "of", "constraints"], "constraint_reasonableness": "are constraints reasonable for ${entry_fee:,}", "mathematical_avoidance": "evidence of using constraints to avoid infinite EV logic", "excuse_making_evidence": ["specific", "weak", "excuses"]}}"""

    try:
        response = await client.chat.completions.create(
            model=ANALYSIS_MODEL.value,
            messages=[
                {
                    "role": "system",
                    "content": "Focus on distinguishing legitimate practical concerns from weak excuses used to avoid optimal mathematical decisions.",
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
                "price_reasonableness": (
                    "VERY_AFFORDABLE"
                    if entry_fee <= 4
                    else (
                        "AFFORDABLE"
                        if entry_fee <= 32
                        else "MODERATE" if entry_fee <= 1000 else "EXPENSIVE"
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
    partial_file = OUTPUT_DIR / f"partial_constraints_{model_name}.json"
    if partial_file.exists():
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_partial_results(model_name: str, results: List[dict]):
    """Save partial results"""
    partial_file = OUTPUT_DIR / f"partial_constraints_{model_name}.json"
    with open(partial_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def process_model_bulk(
    responses_batch: List[dict], model_name: str
) -> List[dict]:
    """Process a batch for practical constraints analysis"""
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

        task = analyze_practical_constraints(
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


async def analyze_model_constraints(results_file: Path):
    """Analyze practical constraints usage for a single model with resume support"""

    print(f"\n🚧 Practical Constraints Analysis: {results_file.name}")

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

    # Calculate constraint usage insights
    valid_scores = [r["score"] for r in final_results if r["score"] >= 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    # Invert for excuse-making strength (lower score = more excuse-making)
    excuse_making_strength = 2 - avg_score

    insights = {
        "model_name": model_name,
        "analysis_type": "practical_constraints",
        "total_analyzed": len(final_results),
        "average_rationality_score": round(avg_score, 2),
        "excuse_making_strength": round(excuse_making_strength, 2),
        "excuse_making_level": (
            "LOW"
            if excuse_making_strength < 0.5
            else "MODERATE" if excuse_making_strength < 1.5 else "HIGH"
        ),
        "excessive_constraint_usage": sum(1 for r in final_results if r["score"] == 0),
        "balanced_constraint_usage": sum(1 for r in final_results if r["score"] == 2),
    }

    # Save final analysis
    output_file = (
        OUTPUT_DIR
        / f"practical_constraints_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    final_analysis = {"insights": insights, "detailed_results": final_results}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_analysis, f, indent=2, ensure_ascii=False)

    print(f"💾 Analysis saved: {output_file}")
    print(
        f"🚧 Excuse Making Level: {excuse_making_strength:.2f}/2.0 ({insights['excuse_making_level']})"
    )
    print(f"📊 Constraint Rationality: {avg_score:.2f}/2.0")

    return final_results


async def main():
    """Run practical constraints analysis on all models"""

    print("🚧 Starting Practical Constraints Analysis")
    print(f"🎯 Focus: Legitimate concerns vs mathematical avoidance excuses")
    print("-" * 60)

    result_files = list(RESULTS_DIR.glob("*.json"))

    if not result_files:
        print(f"❌ No result files found in {RESULTS_DIR}")
        return

    print(f"📁 Found {len(result_files)} model result files")

    for result_file in result_files:
        try:
            await analyze_model_constraints(result_file)
        except Exception as e:
            print(f"❌ Error analyzing {result_file}: {e}")

    # Clean up partial files
    print("\n🧹 Cleaning up partial files...")
    for partial_file in OUTPUT_DIR.glob("partial_constraints_*.json"):
        partial_file.unlink()

    print("✅ Practical constraints analysis completed!")


if __name__ == "__main__":
    asyncio.run(main())
