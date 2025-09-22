#!/usr/bin/env python3
"""
Run ambiguity game with opportunity hunter persona on all models
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from ambiguity_game import main as run_ambiguity_game
from models_enum import AzureModels

# Models to test
MODELS = [
    AzureModels.GPT_4_1,
    AzureModels.GPT_4O,
    AzureModels.GPT_4O_MINI,
    AzureModels.GPT_5,
]


async def run_all_models():
    """Run ambiguity game with opportunity hunter persona on all models"""

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"🎯 Running Opportunity Hunter analysis on {model.value}")
        print(f"{'='*60}")

        # Temporarily modify the MODEL in ambiguity_game
        import ambiguity_game as ag

        original_model = ag.MODEL
        ag.MODEL = model

        try:
            await run_ambiguity_game()
            print(f"✅ Completed {model.value}")
        except Exception as e:
            print(f"❌ Error with {model.value}: {e}")
        finally:
            # Restore original model
            ag.MODEL = original_model

    print(f"\n{'='*60}")
    print("🎉 All models completed!")
    print("📁 Results stored in: results/opportunity_hunter/")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(run_all_models())
