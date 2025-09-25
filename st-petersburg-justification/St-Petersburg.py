# -*- coding: utf-8 -*-
"""
Finance Analysis Script
=======================

This script automates experiments with the St. Petersburg paradox game using
various LLMs (Azure OpenAI). It parses model outputs, aggregates decisions,
performs keyword analysis, and generates visualizations and CSV exports.

------------------------
USER SETUP REQUIRED
------------------------
1. Install dependencies:
   pip install openai pandas matplotlib seaborn tqdm

2. Set environment variables for sensitive data:
   export AZURE_OPENAI_API_KEY="your_api_key_here"
   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
   export AZURE_OPENAI_API_VERSION="2024-12-01-preview"

3. Configure file paths in the CONFIGURATION section below.
"""

# ============= IMPORTS =============
import os
import re
import csv
import json
import itertools
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from openai import AzureOpenAI

# ============= CONFIGURATION =============
# TODO: Set your local or cloud storage paths here
INPUT_FILE_PATH = "/content/drive/My Drive/Finance/st_petersburg_results.txt"   # Raw text results
OUTPUT_CSV_PATH = "/content/drive/My Drive/Finance/st_petersburg_results.csv"   # Structured CSV
OUTPUT_CLEAN_CSV_PATH = "/content/drive/My Drive/Finance/st_petersburg_results_clean.csv"
OUTPUT_HEATMAP_CSV = "/content/drive/My Drive/Finance/heatmap_play_probabilities.csv"
OUTPUT_COOC_CSV = "/content/drive/My Drive/Finance/keyword_cooccurrence.csv"

# Models and experimental parameters
ENTRY_FEES: List[int] = [1, 2, 4, 8, 16, 32, 100, 1000, 10000, 100000]
MODELS: List[str] = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5"]
RUNS_PER_MODEL: int = 10

# Azure OpenAI API settings
AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Validate configuration
if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Missing Azure OpenAI configuration. "
                     "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT as environment variables.")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============= PROMPT TEMPLATE =============
PROMPT_TEMPLATE = """Imagine you are an expert agent in St. Petersburg paradox game...
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
}}
"""

# ============= UTILITY FUNCTIONS =============

def extract_balanced_json(s: str) -> Optional[str]:
    """Extract the first balanced {...} JSON-like substring from a string."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def try_load_json(s: str) -> Optional[Dict[str, Any]]:
    """Try loading a JSON string with minor fixes if parsing fails."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        fixed = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        fixed = re.sub(r",\s*([\}\]])", r"\1", fixed)  # remove trailing commas
        candidate = extract_balanced_json(fixed) or fixed
        try:
            return json.loads(candidate)
        except Exception:
            return None


def parse_log_file(filepath: str) -> pd.DataFrame:
    """Parse raw experiment log into structured DataFrame."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"===== Results for (.*?) =====", content)
    for i in range(1, len(sections), 2):
        model_name = sections[i].strip()
        section_text = sections[i + 1]

        run_pattern = r"Run\s+(\d+),\s*Fee\s+(\d+):\s*(?:```json)?\s*(\{.*?\})(?:```)?"
        for match in re.finditer(run_pattern, section_text, re.DOTALL):
            run = int(match.group(1))
            fee = int(match.group(2))
            json_str = match.group(3)

            try:
                obj = json.loads(json_str)
                decision = obj.get("Decision", "").strip()
                justification = obj.get("Justification", "").strip()
            except Exception:
                decision, justification = "Error", ""

            data.append({
                "model": model_name,
                "run": run,
                "fee": fee,
                "decision": decision,
                "justification": justification
            })

    return pd.DataFrame(data)


def run_experiments():
    """Run experiments with Azure OpenAI and save outputs."""
    client = AzureOpenAI(api_version=AZURE_OPENAI_VERSION,
                         azure_endpoint=AZURE_OPENAI_ENDPOINT,
                         api_key=AZURE_OPENAI_KEY)

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csv_out, \
         open(INPUT_FILE_PATH, "w", encoding="utf-8") as txt_out:
        writer = csv.DictWriter(csv_out, fieldnames=["Model", "Run", "Entrance Fee", "Decision", "Justification"])
        writer.writeheader()

        for model in tqdm(MODELS, desc="Models"):
            txt_out.write(f"\n===== Results for {model} =====\n")
            for run in range(1, RUNS_PER_MODEL + 1):
                for fee in tqdm(ENTRY_FEES, leave=False, desc=f"{model} Run {run}"):
                    prompt = PROMPT_TEMPLATE.format(entrance_fee=fee)
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        output_text = response.choices[0].message.content.strip()
                        txt_out.write(f"Run {run}, Fee {fee}: {output_text}\n")

                        try:
                            output_json = json.loads(output_text)
                            row = {
                                "Model": model,
                                "Run": run,
                                "Entrance Fee": output_json.get("Entrance Fee", fee),
                                "Decision": output_json.get("Decision", ""),
                                "Justification": output_json.get("Justification", "")
                            }
                        except json.JSONDecodeError:
                            row = {"Model": model, "Run": run, "Entrance Fee": fee,
                                   "Decision": "PARSE_ERROR", "Justification": output_text}
                        writer.writerow(row)

                    except Exception as e:
                        logger.error("Request failed: %s", e)
                        writer.writerow({"Model": model, "Run": run, "Entrance Fee": fee,
                                         "Decision": "ERROR", "Justification": str(e)})


# ============= MAIN SCRIPT =============
if __name__ == "__main__":
    logger.info("Starting experiments...")
    run_experiments()
    logger.info("Experiments complete. Parsing results...")

    df = parse_log_file(INPUT_FILE_PATH)
    logger.info("Parsed %d records.", len(df))

    # Example visualization (more can be added following the same pattern)
    plt.figure(figsize=(10, 6))
    for model in df["model"].unique():
        probs = []
        for fee in ENTRY_FEES:
            subset = df[(df["model"] == model) & (df["fee"] == fee)]
            p_play = (subset["decision"] == "Play").mean() if len(subset) else 0
            probs.append(p_play)
        plt.plot(ENTRY_FEES, probs, marker="o", label=model)

    plt.xscale("log")
    plt.xticks(ENTRY_FEES, [f"{fee:,}" for fee in ENTRY_FEES])
    plt.xlabel("Entry Fee ($)")
    plt.ylabel("P(Play)")
    plt.title("Probability of 'Play' vs Entry Fee")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()