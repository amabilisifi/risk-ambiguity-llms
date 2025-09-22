import asyncio
import json
import os
import sys
from enum import Enum

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROBABILITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

SAFE_AMOUNTS = [50, 100, 150]

EV_MULTIPLIERS = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]  # From risk-averse to risk-seeking


def generate_scenarios():
    scenarios = []
    scenario_id = 1

    for safe_amount in SAFE_AMOUNTS:
        for probability in PROBABILITIES:
            for ev_multiplier in EV_MULTIPLIERS:
                # Calculate risky reward to achieve target EV ratio
                target_ev = safe_amount * ev_multiplier
                risky_reward = (
                    round(target_ev / probability / 10) * 10
                )  

                # Skip scenarios with unreasonably large payoffs
                if risky_reward > 2000:  
                    continue

                # Calculate actual values
                actual_ev_risky = probability * risky_reward
                actual_ev_ratio = actual_ev_risky / safe_amount

                # Skip scenarios where risky reward is too close to safe amount
                if risky_reward <= safe_amount * 1.1:  
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

    return scenarios


SCENARIOS = generate_scenarios()

print(f"📊 Generated {len(SCENARIOS)} scenarios:")
print(f"   • Safe amounts: {SAFE_AMOUNTS}")
print(f"   • Probabilities: {PROBABILITIES}")
print(
    f"   • EV ratios range: {min(s['ev_ratio'] for s in SCENARIOS):.2f} to {max(s['ev_ratio'] for s in SCENARIOS):.2f}"
)
print(f"   • Max risky reward: {max(s['risky_reward'] for s in SCENARIOS)}")

if os.environ.get("SKIP_AZURE") != "1":
    from dotenv import load_dotenv
    from models import get_async_client
    from models_enum import AzureModels

    load_dotenv()
    client = get_async_client()
else:
    client = None


async def ask_model_async(
    scenario: dict, model_name: str, trial_number: int, max_retries: int = 3
) -> dict:
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

            response = await asyncio.wait_for(api_call, timeout=120.0)

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
            print(
                f"⏰ Timeout on scenario {scenario['id']} for {model_name} (attempt {attempt + 1}/{max_retries})"
            )
            if attempt == max_retries - 1: 
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
            print(
                f"❌ Error in scenario {scenario['id']} for {model_name} (attempt {attempt + 1}): {e}"
            )
            if attempt == max_retries - 1: 
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


async def run_trials_for_model(model_name: str, n_trials: int = 5):

    safe_model_name = model_name.replace(".", "-").replace("/", "-")
    timestamp = asyncio.get_event_loop().time()
    out_file = f"./results/{safe_model_name}_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print(f"\n🚀 Starting OPPORTUNITY HUNTER trials for {model_name}")
    print(f"   • Using risk-seeking persona intervention")
    print(
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

    # Process in batches of 40
    batch_size = 40

    for batch_start in range(0, total_tasks, batch_size):
        batch_end = min(batch_start + batch_size, total_tasks)
        batch_tasks = all_tasks[batch_start:batch_end]

        print(
            f"\n🔄 Processing batch {batch_start//batch_size + 1}/{(total_tasks + batch_size - 1)//batch_size}"
        )
        print(f"   📦 Requests {batch_start + 1}-{batch_end} of {total_tasks}")

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
                    print(f"   ❌ Batch task failed: {result}")
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
            print(f"   ❌ Batch failed: {e}")
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
        print(
            f"   📊 Progress: {completed}/{total_tasks} ({progress:.1f}%) - Errors: {errors}"
        )

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if batch_end < total_tasks:
            await asyncio.sleep(1)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    success_rate = ((total_tasks - errors) / total_tasks) * 100
    print(f"✅ {model_name} completed: {success_rate:.1f}% success rate")
    print(f"   📁 Results saved to: {out_file}")

    return out_file, {
        "total": total_tasks,
        "errors": errors,
        "success_rate": success_rate,
    }


async def main():
    selected_models = [
        AzureModels.O3_MINI.value,
    ]

    n_trials = 5
    total_requests = len(selected_models) * len(SCENARIOS) * n_trials

    print("🎯 OPPORTUNITY HUNTER PERSONA EXPERIMENT")
    print("=" * 60)
    print(f"📋 EXPERIMENT DESIGN:")
    print(f"   • Scenarios: {len(SCENARIOS)} carefully balanced choices")
    print(f"   • Models: {len(selected_models)} AI systems")
    print(f"   • Trials per scenario: {n_trials}")
    print(f"   • Total API calls: {total_requests:,}")
    print(
        f"   • Estimated duration: {total_requests // 60:.0f}-{total_requests // 30:.0f} minutes"
    )
    print()
    print(f"🔬 SCENARIO CHARACTERISTICS:")
    print(f"   • Safe amounts: {SAFE_AMOUNTS}")
    print(f"   • Win probabilities: {[int(p*100) for p in PROBABILITIES]}%")
    print(
        f"   • EV ratios: {min(s['ev_ratio'] for s in SCENARIOS):.1f} to {max(s['ev_ratio'] for s in SCENARIOS):.1f}"
    )
    print()

    # Run experiments
    all_results = {}
    start_time = asyncio.get_event_loop().time()

    for i, model in enumerate(selected_models, 1):
        print(f"🔄 MODEL {i}/{len(selected_models)}: {model}")

        try:
            result_file, stats = await run_trials_for_model(model, n_trials)
            all_results[model] = {"file": result_file, "stats": stats}
        except Exception as e:
            print(f"❌ Failed to process {model}: {e}")
            all_results[model] = {"error": str(e)}

    # Final summary
    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print(f"   ⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"   📁 Results directory: ./results/")
    print()
    print("📊 MODEL SUMMARY:")

    for model, data in all_results.items():
        if "error" in data:
            print(f"   ❌ {model}: Failed ({data['error']})")
        else:
            stats = data["stats"]
            print(
                f"   ✅ {model}: {stats['success_rate']:.1f}% success ({stats['total']-stats['errors']}/{stats['total']})"
            )

    print(f"\n🔬 Ready for CRRA analysis!")
    print(f"   Next step: python fit_crra_proper.py")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
