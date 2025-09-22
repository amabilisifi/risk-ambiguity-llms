
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Import required libraries
import re
import json
import csv
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm

try:
    from models import get_async_client
    from models_enum import AzureModels
except ImportError:
    print("Warning: Could not import models module. Using direct Azure OpenAI client.")
    from openai import AzureOpenAI

def create_st_petersburg_prompt(entry_fee: int) -> str:
    """Create the detailed prompt explaining the St. Petersburg Paradox game"""

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

**MATHEMATICAL EXPECTED VALUE:**
Expected Value = (1/2 × $2) + (1/4 × $4) + (1/8 × $8) + (1/16 × $16) + ...
Expected Value = $1 + $1 + $1 + $1 + ... = INFINITE DOLLARS

**YOUR DECISION:**
The entry fee to play this game is ${entry_fee:,}.

You must decide: Will you pay ${entry_fee:,} to play this game, or will you pass?

**REQUIRED RESPONSE FORMAT:**
You must respond ONLY in valid JSON format with exactly these fields:
{{
    "decision": "PLAY" | "PASS",
    "reasoning": "Your complete step-by-step thinking process. Explain your understanding of the infinite expected value, how you evaluate the risk vs reward, what factors influence your decision, and your complete chain of thought. Be very detailed about how you think about infinity, probability, and practical considerations.",
    "mathematical_understanding": "Explain your understanding of why the expected value is infinite and what this means to you",
    "risk_assessment": "Describe how you evaluate the practical risks and realistic outcomes",
    "final_justification": "Summarize why you chose PLAY or PASS given all considerations"
}}

Think step by step and provide your most honest assessment. Consider both the mathematical theory and practical reality.
"""

from google.colab import drive
drive.mount('/content/drive')

import re
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
import itertools

input_file = "st_petersburg_results.txt"

data = []
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split by model sections
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

df = pd.DataFrame(data)

# ================================
# Step 2. Aggregate results
# ================================
summary = (
    df.groupby(["model", "fee", "decision"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# ================================
# Step 3. Keyword Analysis
# ================================
keywords = [
    "risk", "infinity", "infinite", "utility", "expectation", "expected value",
    "payoff", "probability", "rational", "irrational", "diminishing", "bounded",
    "real-world", "mathematical", "paradox", "favorable", "unfavorable", "gamble"
]

keyword_counts = Counter()
model_keyword_counts = defaultdict(Counter)
cooccurrence = defaultdict(Counter)

for _, row in df.iterrows():
    just = row["justification"].lower()
    present = []
    for kw in keywords:
        if kw in just:
            keyword_counts[kw] += 1
            model_keyword_counts[row["model"]][kw] += 1
            present.append(kw)
    # co-occurrence
    for a, b in itertools.combinations(present, 2):
        cooccurrence[a][b] += 1
        cooccurrence[b][a] += 1

# ================================
# Step 4. Visualizations
# ================================

# Fixed entry fees list
entry_fees = [1, 2, 4, 8, 16, 32, 100, 1000, 10000, 100000]

# 1. Play Probability Curve
plt.figure(figsize=(10, 6))
for model in df["model"].unique():
    probs = []
    for fee in entry_fees:
        subset = df[(df["model"] == model) & (df["fee"] == fee)]
        p_play = (subset["decision"] == "Play").mean() if len(subset) else 0
        probs.append(p_play)
    plt.plot(entry_fees, probs, marker="o", label=model)

plt.xscale("log")
plt.xticks(entry_fees, [f"{fee:,}" for fee in entry_fees])  # add commas
plt.xlabel("Entry Fee ($)")
plt.ylabel("P(Play)")
plt.title("Probability of 'Play' vs Entry Fee")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 2. Heatmap of Play Probabilities
heatmap_data = []
models_sorted = sorted(df["model"].unique())
for m in models_sorted:
    row = []
    for fee in entry_fees:
        subset = df[(df["model"] == m) & (df["fee"] == fee)]
        row.append((subset["decision"] == "Play").mean() if len(subset) else 0)
    heatmap_data.append(row)

plt.figure(figsize=(12, 4))
sns.heatmap(heatmap_data, annot=True, xticklabels=[f"{fee:,}" for fee in entry_fees],
            yticklabels=models_sorted, cmap="Blues", cbar_kws={'label': 'P(Play)'})
plt.title("Heatmap of 'Play' Probability per Model and Entry Fee")
plt.xlabel("Entry Fee ($)")
plt.ylabel("Model")
plt.show()

# 3. Keyword Frequency Bar Chart (All models combined)
plt.figure(figsize=(12, 6))
common = keyword_counts.most_common(15)
plt.bar([c[0] for c in common], [c[1] for c in common])
plt.title("Top Keywords in Justifications (All Models)")
plt.xlabel("Keyword")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.show()

# 4. Keyword Mentions per Model (Keywords on X-axis, Models in legend)
df_keywords = pd.DataFrame(model_keyword_counts).fillna(0).T
top_keywords = [kw for kw, _ in keyword_counts.most_common(12)]
df_keywords = df_keywords[top_keywords].fillna(0)

# Transpose so that x = keyword, legend = model
df_keywords.T.plot(kind="bar", figsize=(12, 6))
plt.title("Keyword Mentions per Model")
plt.ylabel("Count")
plt.xlabel("Keyword")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 5. Co-occurrence Heatmap
kw_list = [kw for kw, _ in common[:12]]  # top 12 keywords
matrix = pd.DataFrame([[cooccurrence[a].get(b, 0) for b in kw_list] for a in kw_list],
                      index=kw_list, columns=kw_list)

plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, fmt="g", cmap="YlGnBu")  # fmt="g" fixes sci-notation
plt.title("Keyword Co-occurrence in Justifications")
plt.show()

# ================================
# Step 5. Export CSVs
# ================================

# 1. Heatmap data (Play Probabilities per Model and Fee)
heatmap_df = pd.DataFrame(
    heatmap_data,
    index=models_sorted,
    columns=fees_sorted
)
heatmap_df.to_csv("/content/drive/My Drive/Finance/heatmap_play_probabilities.csv")

# 2. Co-occurrence matrix
matrix.to_csv("/content/drive/My Drive/Finance/keyword_cooccurrence.csv")

subscription_key = "15d65736dbcc47609d24fbd58af96bae"

import re
import json
import csv

txt_file = "/content/drive/My Drive/Finance/st_petersburg_results.txt"
csv_file = "/content/drive/My Drive/Finance/st_petersburg_results_clean.csv"

def extract_balanced_json(s):
    """Return the first balanced {...} substring from s or None."""
    start = s.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def try_load_json(s):
    """Try to load JSON string; attempt small fixes if initial load fails."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # quick fixes: normalize smart quotes and remove trailing commas before } or ]
        fixed = s.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
        fixed = re.sub(r",\s*([\}\]])", r"\1", fixed)  # remove trailing commas
        # try to extract balanced JSON again
        candidate = extract_balanced_json(fixed) or fixed
        try:
            return json.loads(candidate)
        except Exception:
            return None

# Read entire file
with open(txt_file, "r", encoding="utf-8") as f:
    text = f.read()

# Split into (model, block) pairs. parts: [pretext, model1, block1, model2, block2, ...]
parts = re.split(r"^=+\s*Results for\s*(.+?)\s*=+\s*$", text, flags=re.MULTILINE)
rows = []

for i in range(1, len(parts), 2):
    model = parts[i].strip()
    block = parts[i+1]

    # find all Run headers inside the model block
    run_pattern = re.compile(r"^\s*Run\s+(\d+),\s*Fee\s+(\d+):", re.MULTILINE)
    matches = list(run_pattern.finditer(block))
    if not matches:
        continue

    for idx, m in enumerate(matches):
        run_num = int(m.group(1))
        fee_from_header = int(m.group(2))
        start = m.end()
        end = matches[idx+1].start() if idx+1 < len(matches) else len(block)
        content = block[start:end].strip()

        # Remove surrounding code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)

        # Try to extract balanced JSON then parse
        json_text = extract_balanced_json(content)
        parsed = None
        if json_text:
            parsed = try_load_json(json_text)
        else:
            # Maybe JSON is inline without braces (rare) or malformed: try whole content
            parsed = try_load_json(content)

        if parsed:
            # handle possible key variants (case / small typos)
            entry_fee = (parsed.get("Entrance Fee")
                         or parsed.get("Entrance fee")
                         or parsed.get("entrance_fee")
                         or fee_from_header)
            decision = (parsed.get("Decision")
                        or parsed.get("decision")
                        or parsed.get("Decision ")
                        or "")
            justification = (parsed.get("Justification")
                             or parsed.get("justification")
                             or parsed.get("Reason")
                             or "")
            rows.append({
                "Model": model,
                "Run": run_num,
                "Entrance Fee": entry_fee,
                "Decision": decision,
                "Justification": justification
            })
        else:
            # Couldn't parse — keep raw content for debugging
            rows.append({
                "Model": model,
                "Run": run_num,
                "Entrance Fee": fee_from_header,
                "Decision": "PARSE_ERROR",
                "Justification": content
            })

# Write CSV
with open(csv_file, "w", newline="", encoding="utf-8") as outf:
    fieldnames = ["Model", "Run", "Entrance Fee", "Decision", "Justification"]
    writer = csv.DictWriter(outf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} rows to {csv_file}")

import json
import csv
from openai import AzureOpenAI
from tqdm import tqdm

# Azure client setup
endpoint = "https://zanistagpteastus2.openai.azure.com/"
api_version = "2024-12-01-preview"
client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=subscription_key)

# Entry fees and models
entry_fees = [1, 2, 4, 8, 16, 32, 100, 1000, 10000, 100000]
models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5"]  # Added gpt-4o-mini

# Prompt template
prompt_template = """Imagine you are an expert agent in St. Petersburg paradox game and offered to play a game.
Your task is to determine and decide whether you pay {entrance_fee} dollars to play this game or will you pass

Here are the rules of this game:
    1. The game starts with $2 in the pot.
    2. A fair coin is flipped repeatedly until it lands HEADS for the first time.
    3. Each time the coin lands TAILS, the pot doubles (2 → 4 → 8 → 16 → 32...).
    4. When the coin finally lands HEADS, the game ends and you win whatever amount is in the pot.
    5. The coin could potentially be flipped for a long time if it keeps landing tails.

    POSSIBLE OUTCOMES & PAYOFFS:
    - Heads on 1st flip: Win $2 (probability = 1/2)
    - Tails then Heads: Win $4 (probability = 1/4)
    - Tails-Tails then Heads: Win $8 (probability = 1/8)
    - Tails-Tails-Tails then Heads: Win $16 (probability = 1/16)
    - And so on...

The output format should be a valid JSON object as follows.
The only thing you should write is this object and nothing else. Do NOT write anything else other than a valid JSON object with this structure:
{{
    "Entrance Fee": {entrance_fee},
    "Decision": "Play/Pass",
    "Justification": "What is your reason? Why did you make this decision?"
}}
"""

# Output files
txt_file = "/content/drive/My Drive/Finance/st_petersburg_results.txt"
csv_file = "/content/drive/My Drive/Finance/st_petersburg_results.csv"

# Open CSV file for structured storage
with open(csv_file, "w", newline="", encoding="utf-8") as csv_out, open(txt_file, "w", encoding="utf-8") as txt_out:
    writer = csv.DictWriter(csv_out, fieldnames=["Model", "Run", "Entrance Fee", "Decision", "Justification"])
    writer.writeheader()

    for model in tqdm(models):
        txt_out.write(f"\n===== Results for {model} =====\n")
        for run in range(1, 11):  # 10 runs
            for fee in tqdm(entry_fees):
            #for run in range(1, 11):  # 10 runs
                prompt = prompt_template.format(entrance_fee=fee)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                output_text = response.choices[0].message.content.strip()

                # Write raw text output to TXT
                txt_out.write(f"Run {run}, Fee {fee}: {output_text}\n")
                print(f"{model} | Run {run}, Fee {fee}: {output_text}")

                # Try to parse JSON
                try:
                    output_json = json.loads(output_text)
                    row = {
                        "Model": model,
                        "Run": run,
                        "Entrance Fee": output_json.get("Entrance Fee", fee),
                        "Decision": output_json.get("Decision", ""),
                        "Justification": output_json.get("Justification", "")
                    }
                    writer.writerow(row)
                except json.JSONDecodeError:
                    # If output is not valid JSON, log it as-is
                    writer.writerow({
                        "Model": model,
                        "Run": run,
                        "Entrance Fee": fee,
                        "Decision": "PARSE_ERROR",
                        "Justification": output_text
                    })