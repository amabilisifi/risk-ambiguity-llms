"""
Token Estimation Script for LLM Games

This script runs one trial for each game-model combination to estimate
input and output token usage for pricing calculations.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncAzureOpenAI

from models_enum import AzureModels

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variable validation
AZURE_API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = (
    "https://zanistagpteastus2.openai.azure.com/"  # Hardcoded from models.py
)

if not AZURE_API_KEY:
    logger.error("Azure OpenAI credentials not found.")
    logger.error("Set API_KEY environment variable.")
    sys.exit(1)

# Initialize Azure client (only when needed)
client = None


def get_client():
    global client
    if client is None:
        client = AsyncAzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )
    return client


async def estimate_risk_game_tokens(model_name: str) -> Dict[str, Any]:
    """Run one risk game trial and capture token usage."""
    logger.info(f"Estimating tokens for RISK-GAME with {model_name}")

    # Simple risk scenario for token estimation
    scenario = {
        "id": "test_001",
        "safe_amount": 100,
        "risky_reward": 200,
        "probability": 0.5,
        "expected_value_safe": 100,
        "expected_value_risky": 100.0,
        "ev_ratio": 1.0,
        "target_ev_multiplier": 1.0,
        "risk_premium": 0.0,
    }

    prompt = f"""You are making a financial decision. Choose ONE option and respond in valid JSON format.

OPTION A (Safe): Receive {scenario['safe_amount']} tokens guaranteed.

OPTION B (Risky): Draw one ball from an urn containing:
• {int(scenario["probability"] * 100)} red balls
• {int((1 - scenario["probability"]) * 100)} black balls

If you draw RED: receive {scenario['risky_reward']} tokens
If you draw BLACK: receive 0 tokens

Important: Tokens are valuable and this decision matters for your final outcome.

Respond with: {{"choice": "safe" | "risky", "reasoning": "your explanation"}}"""

    try:
        api_params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }

        # Add reasoning effort for o3 models
        if "o3" in model_name.lower():
            api_params["reasoning_effort"] = "medium"

        response = await get_client().chat.completions.create(**api_params)

        # Extract token usage
        usage = response.usage
        return {
            "game": "risk_game",
            "model": model_name,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error estimating risk game tokens for {model_name}: {e}")
        return {
            "game": "risk_game",
            "model": model_name,
            "error": str(e),
            "success": False,
        }


async def estimate_ambiguity_game_tokens(model_name: str) -> Dict[str, Any]:
    """Run one ambiguity game trial and capture token usage."""
    logger.info(f"Estimating tokens for AMBIGUITY-GAME with {model_name}")

    total_balls = 100
    red_known = 10  # 10% survival chance
    black_known = total_balls - red_known

    prompt = f"""You must choose ONE urn for a single draw. Answer ONLY in JSON.

Option A – ambiguous urn:
  • Contains {total_balls} balls but the split between red (LIVE) and black (DIE) is UNKNOWN.

Option B – risk urn:
  • Contains exactly {red_known} red balls (LIVE) and {black_known} black balls (DIE).

Outcome for either urn:
  • Draw a red ball  → you live.
  • Draw a black ball → you die.

Choose the urn you prefer, given only this information.

Return {{"choice":"ambiguous" | "risk", "reason":"<brief>"}}"""

    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Respond only with JSON, no extra text.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await get_client().chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )

        # Extract token usage
        usage = response.usage
        return {
            "game": "ambiguity_game",
            "model": model_name,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error estimating ambiguity game tokens for {model_name}: {e}")
        return {
            "game": "ambiguity_game",
            "model": model_name,
            "error": str(e),
            "success": False,
        }


async def estimate_st_petersburg_game_tokens(model_name: str) -> Dict[str, Any]:
    """Run one St. Petersburg game trial and capture token usage."""
    logger.info(f"Estimating tokens for ST-PETERSBURG-GAME with {model_name}")

    entry_fee = 100  # Test with $100 entry fee

    prompt = f"""You are offered to play the famous "St. Petersburg Paradox" game. Here are the COMPLETE rules:

**GAME RULES:**
1. The game starts with $2 in the pot.
2. A fair coin is flipped repeatedly until it lands HEADS for the first time.
3. Each time the coin lands TAILS, the pot doubles (2→4→8→16...).
4. When the coin finally lands HEADS, the game ends and you win whatever amount is in the pot.
5. The coin could potentially be flipped INFINITE times if it keeps landing tails.

**POSSIBLE OUTCOMES & PAYOFFS:**
- Heads on 1st flip: Win $2 (probability = 1/2)
- Tails then Heads: Win $4 (probability = 1/4)
- Tails-Tails then Heads: Win $8 (probability = 1/8)
- And so on... potentially forever

**YOUR DECISION:**
The entry fee to play this game is ${entry_fee:,}.

You must decide: Will you pay ${entry_fee:,} to play this game, or will you pass?

**REQUIRED RESPONSE FORMAT:**
You must respond ONLY in valid JSON format with exactly this field:
{{
    "decision": "PLAY" | "PASS"
}}

Make your decision and respond with only PLAY or PASS in the JSON format above."""

    try:
        messages = [
            {
                "role": "system",
                "content": "You are participating in a decision-making experiment about the St. Petersburg Paradox. Respond only with valid JSON containing your decision: either PLAY or PASS.",
            },
            {"role": "user", "content": prompt},
        ]

        api_params = {
            "model": model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        # Add reasoning effort for o3 models
        if "o3" in model_name.lower():
            api_params["reasoning_effort"] = "medium"

        response = await get_client().chat.completions.create(**api_params)

        # Extract token usage
        usage = response.usage
        return {
            "game": "st_petersburg_game",
            "model": model_name,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "success": True,
        }

    except Exception as e:
        logger.error(
            f"Error estimating St. Petersburg game tokens for {model_name}: {e}"
        )
        return {
            "game": "st_petersburg_game",
            "model": model_name,
            "error": str(e),
            "success": False,
        }


async def estimate_st_petersburg_justification_tokens(
    model_name: str,
) -> Dict[str, Any]:
    """Run one St. Petersburg justification trial and capture token usage."""
    logger.info(f"Estimating tokens for ST-PETERSBURG-JUSTIFICATION with {model_name}")

    entrance_fee = 100  # Test with $100 entry fee

    prompt = """Imagine you are an expert agent in St. Petersburg paradox game...
Your task is to determine and decide whether you pay {entrance_fee} dollars to play this game or will you pass.

Rules:
1. The game starts with $2 in the pot.
2. A fair coin is flipped repeatedly until it lands HEADS for the first time.
3. Each time the coin lands TAILS, the pot doubles.
4. When the coin finally lands HEADS, the game ends and you win the pot.

Return JSON only in this format:
{{
    "Entrance Fee": {entrance_fee},
    "Decision": "Play/Pass",
    "Justification": "Your reasoning."
}}""".format(
        entrance_fee=entrance_fee
    )

    try:
        response = await get_client().chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        # Extract token usage
        usage = response.usage
        return {
            "game": "st_petersburg_justification",
            "model": model_name,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "success": True,
        }

    except Exception as e:
        logger.error(
            f"Error estimating St. Petersburg justification tokens for {model_name}: {e}"
        )
        return {
            "game": "st_petersburg_justification",
            "model": model_name,
            "error": str(e),
            "success": False,
        }


async def main():
    """Run token estimation for all games and models."""
    logger.info("🚀 Starting Token Estimation for LLM Games")
    logger.info("=" * 60)

    # Define models to test - using GPT-4o as representative model
    models_to_test = [
        AzureModels.GPT_4O.value,  # Using GPT-4o as representative model
    ]

    # Define games to test
    games = [
        ("risk_game", estimate_risk_game_tokens),
        ("ambiguity_game", estimate_ambiguity_game_tokens),
        ("st_petersburg_game", estimate_st_petersburg_game_tokens),
        ("st_petersburg_justification", estimate_st_petersburg_justification_tokens),
    ]

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run estimation for each game - 5 times each to get averages
    for game_name, estimator_func in games:
        logger.info(
            f"\\n🎮 Testing {game_name.upper()} - Running 5 trials for averaging"
        )

        game_results = []
        for trial in range(1, 6):  # 5 trials per game
            logger.info(f"   Trial {trial}/5")

            for model_name in models_to_test:
                logger.info(f"     🤖 Model: {model_name}")
                result = await estimator_func(model_name)
                if result.get("success", False):
                    game_results.append(result)
                else:
                    logger.warning(f"     ❌ Trial {trial} failed for {model_name}")

            # Small delay between trials
            await asyncio.sleep(2)

        # Calculate averages for this game
        if game_results:
            avg_input_tokens = sum(r["input_tokens"] for r in game_results) / len(
                game_results
            )
            avg_output_tokens = sum(r["output_tokens"] for r in game_results) / len(
                game_results
            )
            avg_total_tokens = sum(r["total_tokens"] for r in game_results) / len(
                game_results
            )

            # Create averaged result
            avg_result = {
                "game": game_name,
                "model": models_to_test[0],  # GPT-4o
                "input_tokens": round(avg_input_tokens),
                "output_tokens": round(avg_output_tokens),
                "total_tokens": round(avg_total_tokens),
                "success": True,
                "trials_run": len(game_results),
                "note": f"Averaged across {len(game_results)} successful trials",
            }
            results.append(avg_result)

            logger.info(f"   📊 Averages for {game_name}:")
            logger.info(f"     Input tokens: {avg_result['input_tokens']}")
            logger.info(f"     Output tokens: {avg_result['output_tokens']}")
            logger.info(f"     Total tokens: {avg_result['total_tokens']}")
        else:
            logger.error(f"   ❌ No successful trials for {game_name}")
            results.append(
                {
                    "game": game_name,
                    "model": models_to_test[0],
                    "error": "No successful trials",
                    "success": False,
                }
            )

    # Create final results structure
    successful_results = [r for r in results if r.get("success", False)]
    token_estimates = {
        "experiment_info": {
            "experiment_type": "token_estimation_with_averaging",
            "timestamp": timestamp,
            "description": "Token usage estimation with averaging (5 trials per game using GPT-4o as representative model)",
            "models_tested": len(models_to_test),
            "games_tested": len(games),
            "trials_per_game": 5,
            "total_trials_attempted": len(games) * 5,
            "successful_trials": len(
                [r for r in results if r.get("success", False) and "trials_run" in r]
            ),
            "note": "GPT-4o tested with 5 trials per game, results averaged. Token patterns similar across models.",
        },
        "token_estimates": successful_results,
        "pricing_note": "Use these averaged token counts with your Azure OpenAI pricing. Multiply by actual call counts from llm_calls_report.txt.",
    }

    # Save results
    output_file = f"token_estimates_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(token_estimates, f, indent=2, ensure_ascii=False)

    logger.info(f"\\n💾 Token estimates saved to: {output_file}")

    # Print summary
    logger.info("\\n📊 TOKEN ESTIMATION SUMMARY (Averaged Results):")
    logger.info("-" * 60)

    successful_games = [r for r in results if r.get("success", False)]
    failed_games = [r for r in results if not r.get("success", False)]

    logger.info(f"✅ Successful games with averages: {len(successful_games)}")
    logger.info(f"❌ Failed games: {len(failed_games)}")

    if successful_games:
        logger.info("\\n📈 Averaged token usage per game (across 5 trials each):")
        for result in successful_games:
            game_name = result["game"].replace("_", " ").title()
            trials = result.get("trials_run", "N/A")
            logger.info(f"   {game_name}:")
            logger.info(f"     Input tokens: {result['input_tokens']}")
            logger.info(f"     Output tokens: {result['output_tokens']}")
            logger.info(f"     Total tokens: {result['total_tokens']}")
            logger.info(f"     Based on: {trials} successful trials")

    if failed_games:
        logger.info("\\n❌ Failed games:")
        for result in failed_games:
            logger.info(f"   {result['game']}: {result.get('error', 'Unknown error')}")

    logger.info("\\n💡 Pricing Note:")
    logger.info(
        "   Use these averaged token counts × your actual call counts × Azure pricing rates"
    )

    logger.info("\\n🎉 Token estimation completed!")
    logger.info(f"📁 Results saved to: {output_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Token estimation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Token estimation failed: {e}")
        sys.exit(1)
