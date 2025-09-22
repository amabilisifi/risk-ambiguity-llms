import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from models import get_async_client
from models_enum import AzureModels

# ───────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────
ANALYSIS_MODEL = AzureModels.GPT_4O  # GPT-4o for analysis
BULK_SIZE = 10  # Process 10 responses concurrently
TIMEOUT_SECONDS = 120  # 2 minutes timeout per bulk request
# ───────────────────────────────────────────────────────────

load_dotenv()
client = get_async_client()

# Input and output directories
RESULTS_DIR = Path("../chosen_results")
ANALYSIS_OUTPUT_DIR = Path("./analysis_results")
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)

# Research questions focused on BEHAVIORAL INSIGHTS
BEHAVIORAL_RESEARCH_QUESTIONS = [
    {
        "id": "infinity_understanding_vs_behavior",
        "question": "Understanding of Infinity vs. Actual Behavior",
        "prompt": """Analyze this AI model's understanding of infinity and whether it aligns with its decision.

Context: Entry fee ${entry_fee:,}, Decision: {decision}

Key Analysis:
1. Does the model explicitly understand infinite expected value?
2. Does its decision align with this understanding?
3. What's the gap between theory and practice?

Score behavioral consistency (0-2):
- Score 0: Major inconsistency - understands infinite EV but behaves irrationally
- Score 1: Some inconsistency - mixed signals between understanding and action
- Score 2: Good consistency - behavior aligns with stated understanding

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "behavioral consistency analysis", "infinity_understanding": "level of mathematical understanding", "behavioral_alignment": "how well decision matches understanding", "inconsistency_type": "specific type of contradiction if any"}}""",
    },
    {
        "id": "breakpoint_decision_drivers",
        "question": "Breakpoint Decision Parameters Analysis",
        "prompt": """Analyze what drove this AI model's breakpoint decision at ${entry_fee:,}.

Decision: {decision}

Focus on identifying:
1. What specific threshold triggered this decision?
2. Is this a rational breakpoint given infinite EV?
3. What were the primary decision drivers?

Score breakpoint rationality (0-2):
- Score 0: Irrational breakpoint (stopping at tiny amounts despite infinite EV)
- Score 1: Somewhat rational with mixed logic
- Score 2: Rational breakpoint with clear reasoning

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "breakpoint analysis", "primary_drivers": ["list", "of", "main", "factors"], "breakpoint_rationality": "assessment of decision logic", "threshold_mentioned": "any specific threshold mentioned"}}""",
    },
    {
        "id": "loss_aversion_analysis",
        "question": "Loss Aversion vs. Infinite Expected Value",
        "prompt": """Analyze this AI model's loss aversion behavior vs. infinite expected value understanding.

Entry fee: ${entry_fee:,}, Decision: {decision}

Key questions:
1. Does the model focus more on potential small losses or infinite gains?
2. What evidence of loss aversion bias appears?
3. How does it weigh finite loss vs. infinite expected value?

Score loss aversion impact (0-2):
- Score 0: Extreme loss aversion overriding mathematical logic
- Score 1: Moderate loss aversion with some rational considerations
- Score 2: Balanced approach prioritizing mathematical expected value

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "loss aversion analysis", "loss_focus_evidence": ["specific", "loss", "concerns"], "gain_focus_evidence": ["specific", "gain", "focus"], "mathematical_override": "does loss aversion override infinite EV logic"}}""",
    },
    {
        "id": "probability_distribution_understanding",
        "question": "Understanding of Probability Distribution",
        "prompt": """Analyze how this AI model understands and uses probability distribution knowledge.

Entry fee: ${entry_fee:,}, Decision: {decision}

Focus areas:
1. Does it understand the heavy-tail distribution?
2. How does it weigh small likely outcomes vs. rare large ones?
3. Does it focus on median vs. mean outcomes?

Score distribution understanding (0-2):
- Score 0: Poor understanding of distribution implications
- Score 1: Basic understanding but misapplies it
- Score 2: Good understanding and proper application

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "distribution analysis", "small_outcomes_focus": "how much it focuses on small outcomes", "large_outcomes_consideration": "how it treats rare large outcomes", "median_vs_mean_understanding": "does it distinguish between typical and average outcomes"}}""",
    },
    {
        "id": "utility_function_application",
        "question": "Utility Function and Diminishing Marginal Utility",
        "prompt": """Analyze how this AI model applies utility theory and diminishing marginal utility.

Entry fee: ${entry_fee:,}, Decision: {decision}

Key analysis points:
1. Does it mention utility functions or diminishing marginal utility?
2. How does it use utility theory in decision-making?
3. Does it use utility to justify avoiding mathematically optimal decisions?

Score utility application (0-2):
- Score 0: Uses utility theory as excuse to avoid optimal decisions
- Score 1: Basic utility considerations with mixed application
- Score 2: Sophisticated utility analysis supporting rational decisions

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "utility function analysis", "utility_mentions": ["specific", "utility", "concepts"], "utility_justification": "how utility theory is used to justify decision", "rational_utility_application": "whether utility analysis supports or undermines mathematical logic"}}""",
    },
    {
        "id": "practical_constraints_rationality",
        "question": "Practical Constraints vs. Mathematical Avoidance",
        "prompt": """Analyze whether this AI model uses practical constraints rationally or as excuses.

Entry fee: ${entry_fee:,}, Decision: {decision}

Evaluation focus:
1. What practical constraints does it mention?
2. Are these constraints reasonable or excessive for this price?
3. Is it using practical concerns to avoid mathematical decisions?

Score constraint rationality (0-2):
- Score 0: Uses weak practical excuses to avoid mathematical logic
- Score 1: Some reasonable constraints mixed with over-caution
- Score 2: Well-balanced practical considerations

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "practical constraints analysis", "constraints_mentioned": ["list", "of", "constraints"], "constraint_reasonableness": "are the constraints reasonable for this price level", "mathematical_avoidance": "evidence of using constraints to avoid math"}}""",
    },
    {
        "id": "small_outcomes_bias",
        "question": "Focus on Small Outcomes vs. Infinite Expectation",
        "prompt": """Analyze this AI model's focus on small probable outcomes vs. infinite expected value.

Entry fee: ${entry_fee:,}, Decision: {decision}

Analysis points:
1. How much does it focus on $2, $4, $8 outcomes?
2. Does it understand these are just the most likely, not the expectation?
3. How does it balance likely small outcomes vs. infinite expectation?

Score small outcome bias (0-2):
- Score 0: Obsessively focuses on small outcomes, ignoring infinite EV
- Score 1: Mentions both small and large outcomes with some confusion
- Score 2: Properly balances likely outcomes with infinite expectation

Response: {text}

Return JSON with: {{"score": 0-2, "explanation": "small outcomes focus analysis", "small_outcome_emphasis": "how much it emphasizes $2, $4, $8 outcomes", "infinite_ev_recognition": "how it treats the infinite expected value", "balance_assessment": "does it properly balance likely vs. expected outcomes"}}""",
    },
]


async def analyze_single_response_behavioral(
    text: str, question_config: dict, entry_fee: int, decision: str
) -> dict:
    """Analyze a single response for behavioral insights"""

    # Format prompt with decision context
    prompt = question_config["prompt"].format(
        text=text, entry_fee=entry_fee, decision=decision
    )

    try:
        response = await client.chat.completions.create(
            model=ANALYSIS_MODEL.value,
            messages=[
                {
                    "role": "system",
                    "content": f"You are analyzing AI behavioral patterns in the St. Petersburg Paradox. Focus on the GAP between mathematical understanding and actual decisions. Remember: infinite expected value means rational agents should play at ANY finite price, but some models show extreme behavioral biases. Analyze behavioral inconsistencies, not explanation quality.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        analysis = json.loads(response.choices[0].message.content)

        # Add decision context
        analysis.update(
            {
                "question_id": question_config["id"],
                "question_title": question_config["question"],
                "entry_fee": entry_fee,
                "decision": decision,
                "expected_rational_decision": "PLAY",  # Always PLAY for infinite EV
                "behavioral_rationality": decision == "PLAY",
                "analysis_timestamp": datetime.now().isoformat(),
            }
        )

        return analysis

    except Exception as e:
        return {
            "question_id": question_config["id"],
            "question_title": question_config["question"],
            "entry_fee": entry_fee,
            "decision": decision,
            "score": -1,
            "explanation": f"Analysis error: {str(e)}",
            "error": True,
        }


async def process_bulk_behavioral_analysis(
    responses_batch: List[dict], question_config: dict
) -> List[dict]:
    """Process a batch of responses for behavioral analysis"""

    tasks = []
    for response in responses_batch:
        # Combine all reasoning fields
        full_text = " ".join(
            [
                response.get("reasoning", ""),
                response.get("mathematical_understanding", ""),
                response.get("risk_assessment", ""),
                response.get("final_justification", ""),
            ]
        )

        task = analyze_single_response_behavioral(
            full_text, question_config, response["entry_fee"], response["decision"]
        )
        tasks.append(task)

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=TIMEOUT_SECONDS
        )

        # Handle exceptions
        clean_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = {
                    "question_id": question_config["id"],
                    "entry_fee": responses_batch[i]["entry_fee"],
                    "decision": responses_batch[i]["decision"],
                    "score": -1,
                    "explanation": f"Exception: {str(result)}",
                    "error": True,
                }
                clean_results.append(error_result)
            else:
                clean_results.append(result)

        return clean_results

    except asyncio.TimeoutError:
        print(f"⚠️  Timeout processing batch for {question_config['id']}")
        return [
            {
                "question_id": question_config["id"],
                "entry_fee": resp["entry_fee"],
                "decision": resp["decision"],
                "score": -1,
                "explanation": "Timeout during analysis",
                "error": True,
            }
            for resp in responses_batch
        ]


def calculate_behavioral_insights(analysis_results: Dict) -> Dict:
    """Calculate behavioral insights and patterns"""

    detailed_results = analysis_results.get("detailed_results", [])

    # Basic decision statistics
    total_decisions = len(detailed_results)
    play_decisions = sum(1 for r in detailed_results if r.get("decision") == "PLAY")
    pass_decisions = total_decisions - play_decisions

    # Price-level analysis
    price_analysis = {}
    for result in detailed_results:
        price = result["entry_fee"]
        if price not in price_analysis:
            price_analysis[price] = {"play": 0, "pass": 0, "total": 0}

        price_analysis[price]["total"] += 1
        if result["decision"] == "PLAY":
            price_analysis[price]["play"] += 1
        else:
            price_analysis[price]["pass"] += 1

    # Find breakpoint
    breakpoint_price = None
    for price in sorted(price_analysis.keys()):
        play_rate = price_analysis[price]["play"] / price_analysis[price]["total"]
        if play_rate < 0.5:  # First price where model mostly refuses
            breakpoint_price = price
            break

    # Behavioral scoring analysis
    behavioral_scores = {}
    for question_id, question_data in analysis_results.get(
        "questions_analyzed", {}
    ).items():
        scores = [
            r["score"] for r in question_data["detailed_results"] if r["score"] >= 0
        ]
        behavioral_scores[question_id] = {
            "average_score": sum(scores) / len(scores) if scores else 0,
            "score_distribution": {
                "rational": sum(1 for s in scores if s == 2),
                "mixed": sum(1 for s in scores if s == 1),
                "irrational": sum(1 for s in scores if s == 0),
            },
        }

    return {
        "decision_summary": {
            "total_decisions": total_decisions,
            "play_decisions": play_decisions,
            "pass_decisions": pass_decisions,
            "play_percentage": (
                round((play_decisions / total_decisions) * 100, 2)
                if total_decisions > 0
                else 0
            ),
            "behavioral_rationality": (
                "POOR"
                if play_decisions < total_decisions * 0.3
                else "MODERATE" if play_decisions < total_decisions * 0.7 else "GOOD"
            ),
        },
        "breakpoint_analysis": {
            "effective_breakpoint": breakpoint_price,
            "breakpoint_assessment": (
                "EXTREMELY_LOW"
                if breakpoint_price and breakpoint_price <= 4
                else (
                    "MODERATE"
                    if breakpoint_price and breakpoint_price <= 32
                    else "RATIONAL"
                )
            ),
            "price_level_analysis": price_analysis,
        },
        "behavioral_patterns": behavioral_scores,
        "key_insights": {
            "mathematical_understanding_vs_behavior_gap": behavioral_scores.get(
                "infinity_understanding_vs_behavior", {}
            ).get("average_score", 0),
            "loss_aversion_strength": 2
            - behavioral_scores.get("loss_aversion_analysis", {}).get(
                "average_score", 2
            ),  # Inverted score
            "practical_excuse_usage": 2
            - behavioral_scores.get("practical_constraints_rationality", {}).get(
                "average_score", 2
            ),  # Inverted score
            "small_outcome_bias": 2
            - behavioral_scores.get("small_outcomes_bias", {}).get(
                "average_score", 2
            ),  # Inverted score
        },
    }


async def analyze_model_behavioral_patterns(results_file: Path):
    """Analyze behavioral patterns from a single model"""

    print(f"\n🧠 Behavioral Analysis: {results_file.name}")

    # Load model results
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data["experiment_info"]["model_tested"]
    detailed_results = data.get("detailed_results", [])

    print(f"🤖 Model: {model_name}")
    print(f"📊 Total responses: {len(detailed_results)}")

    # Calculate basic behavioral metrics first
    play_count = sum(1 for r in detailed_results if r["decision"] == "PLAY")
    play_percentage = (play_count / len(detailed_results)) * 100

    print(
        f"🎮 Play Rate: {play_count}/{len(detailed_results)} ({play_percentage:.1f}%)"
    )

    # Store complete analysis
    complete_analysis = {
        "model_name": model_name,
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "behavioral_patterns",
        "total_responses_analyzed": len(detailed_results),
        "analysis_model_used": ANALYSIS_MODEL.value,
        "bulk_size": BULK_SIZE,
        "questions_analyzed": {},
        "detailed_results": detailed_results,  # Store original results for behavioral calculation
    }

    # Process each behavioral question
    for question_config in BEHAVIORAL_RESEARCH_QUESTIONS:
        print(f"\n  🔬 Analyzing: {question_config['question']}")

        question_results = []

        # Process in batches
        for i in range(0, len(detailed_results), BULK_SIZE):
            batch = detailed_results[i : i + BULK_SIZE]
            batch_num = (i // BULK_SIZE) + 1
            total_batches = (len(detailed_results) + BULK_SIZE - 1) // BULK_SIZE

            print(
                f"    📦 Processing batch {batch_num}/{total_batches} ({len(batch)} responses)"
            )

            batch_results = await process_bulk_behavioral_analysis(
                batch, question_config
            )
            question_results.extend(batch_results)

        # Calculate statistics
        valid_scores = [r["score"] for r in question_results if r["score"] >= 0]
        error_count = sum(1 for r in question_results if r.get("error", False))

        question_summary = {
            "question_config": question_config,
            "total_analyzed": len(question_results),
            "valid_analyses": len(valid_scores),
            "errors": error_count,
            "average_score": (
                round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0
            ),
            "score_distribution": {
                "score_0": sum(1 for s in valid_scores if s == 0),
                "score_1": sum(1 for s in valid_scores if s == 1),
                "score_2": sum(1 for s in valid_scores if s == 2),
            },
            "detailed_results": question_results,
        }

        complete_analysis["questions_analyzed"][
            question_config["id"]
        ] = question_summary

        print(f"    ✅ Average score: {question_summary['average_score']:.2f}")
        print(
            f"    📊 Distribution: Rational={question_summary['score_distribution']['score_2']}, Mixed={question_summary['score_distribution']['score_1']}, Irrational={question_summary['score_distribution']['score_0']}"
        )

    # Calculate behavioral insights
    behavioral_insights = calculate_behavioral_insights(complete_analysis)
    complete_analysis["behavioral_insights"] = behavioral_insights

    # Save analysis
    output_file = (
        ANALYSIS_OUTPUT_DIR
        / f"behavioral_analysis_{model_name.replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(complete_analysis, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Behavioral analysis saved: {output_file}")

    # Print key insights
    insights = behavioral_insights["key_insights"]
    print(f"\n🧩 Key Behavioral Insights:")
    print(
        f"   📈 Play Rate: {behavioral_insights['decision_summary']['play_percentage']:.1f}% ({behavioral_insights['decision_summary']['behavioral_rationality']})"
    )
    print(
        f"   🔄 Theory-Practice Gap: {insights['mathematical_understanding_vs_behavior_gap']:.2f}/2.0"
    )
    print(f"   😰 Loss Aversion Strength: {insights['loss_aversion_strength']:.2f}/2.0")
    print(
        f"   🎯 Breakpoint: ${behavioral_insights['breakpoint_analysis']['effective_breakpoint']:,} ({behavioral_insights['breakpoint_analysis']['breakpoint_assessment']})"
    )

    return complete_analysis


async def main():
    """Run behavioral analysis on all model result files"""

    print("🧠 Starting St. Petersburg Paradox Behavioral Analysis")
    print(f"🤖 Analysis Model: {ANALYSIS_MODEL.value}")
    print(f"📦 Bulk Size: {BULK_SIZE} concurrent requests")
    print(f"⏱️  Timeout: {TIMEOUT_SECONDS} seconds per bulk")
    print(f"🎯 Focus: Behavioral patterns and decision-making inconsistencies")
    print("-" * 70)

    # Find all result files
    result_files = list(RESULTS_DIR.glob("*.json"))

    if not result_files:
        print(f"❌ No result files found in {RESULTS_DIR}")
        return

    print(f"📁 Found {len(result_files)} model result files")

    all_analyses = {}

    # Process each model separately
    for result_file in result_files:
        try:
            analysis = await analyze_model_behavioral_patterns(result_file)
            all_analyses[analysis["model_name"]] = analysis
        except Exception as e:
            print(f"❌ Error analyzing {result_file}: {e}")

    # Create comparative behavioral analysis
    master_behavioral_summary = {
        "analysis_summary": {
            "total_models_analyzed": len(all_analyses),
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_type": "behavioral_patterns",
            "analysis_model": ANALYSIS_MODEL.value,
            "research_focus": "Behavioral inconsistencies and decision-making patterns",
        },
        "model_behavioral_comparison": {},
        "individual_analyses": all_analyses,
    }

    # Compare behavioral patterns across models
    print(f"\n📊 BEHAVIORAL COMPARISON ACROSS MODELS:")
    print("=" * 70)

    for model_name, analysis in all_analyses.items():
        insights = analysis["behavioral_insights"]
        decision_summary = insights["decision_summary"]
        breakpoint = insights["breakpoint_analysis"]["effective_breakpoint"]

        behavioral_profile = {
            "play_percentage": decision_summary["play_percentage"],
            "behavioral_rationality": decision_summary["behavioral_rationality"],
            "breakpoint": breakpoint,
            "theory_practice_gap": insights["key_insights"][
                "mathematical_understanding_vs_behavior_gap"
            ],
            "loss_aversion": insights["key_insights"]["loss_aversion_strength"],
            "practical_excuse_usage": insights["key_insights"][
                "practical_excuse_usage"
            ],
            "small_outcome_bias": insights["key_insights"]["small_outcome_bias"],
        }

        master_behavioral_summary["model_behavioral_comparison"][
            model_name
        ] = behavioral_profile

        print(f"\n🤖 {model_name.upper()}:")
        print(
            f"   ▶️  Play Rate: {decision_summary['play_percentage']:.1f}% ({decision_summary['behavioral_rationality']})"
        )
        print(
            f"   💰 Breakpoint: ${breakpoint:,} ({insights['breakpoint_analysis']['breakpoint_assessment']})"
        )
        print(
            f"   🧠 Theory-Practice Gap: {behavioral_profile['theory_practice_gap']:.2f}/2.0"
        )
        print(f"   😰 Loss Aversion: {behavioral_profile['loss_aversion']:.2f}/2.0")
        print(
            f"   🎯 Small Outcome Bias: {behavioral_profile['small_outcome_bias']:.2f}/2.0"
        )

    # Save master behavioral summary
    master_file = (
        ANALYSIS_OUTPUT_DIR
        / f"master_behavioral_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(master_behavioral_summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Master behavioral analysis saved: {master_file}")
    print("=" * 70)

    return master_behavioral_summary


if __name__ == "__main__":
    print("🚀 Starting Behavioral Analysis of St. Petersburg Paradox Results")
    print(
        "💡 Focus: Understanding vs. Behavior gaps, Loss aversion, Breakpoint rationality"
    )
    print()

    asyncio.run(main())
