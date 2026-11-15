# Translate the MATH-500 problems into Estonian with Claude Fable 5.2.
# Uses the same translation prompt as Barua et al. (utils/prompts.py).
# Results are saved after every problem. If the script stops, run it again
# and it continues where it left off.

import json
import os
import re
import sys

import anthropic
from datasets import load_from_disk

sys.path.insert(0, "Multilingual-Long-CoT")
from utils.prompts import translate_template  # Barua's translation prompt

MODEL = "claude-fable-5-2"
DATA_DIR = "Multilingual-Long-CoT/eval/benchmarks/mt-math-500"
OUT_FILE = "translated_et_fable.json"
LIMIT = 500  # test with 3 problems first, then change to 500

data = load_from_disk(DATA_DIR)
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# Load earlier results so we do not translate the same problem twice
if os.path.exists(OUT_FILE):
    with open(OUT_FILE, encoding="utf-8") as f:
        done = json.load(f)
else:
    done = []

for nr in range(len(done), LIMIT):
    english = data[nr]["en"]
    prompt = translate_template.format(target_lang="Estonian", source_text=english)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    # Keep only the text parts of the answer
    answer = "".join(block.text for block in response.content if block.type == "text")

    # The prompt asks the model to put the translation inside <translation> tags
    match = re.search(r"<translation>(.*?)</translation>", answer, re.DOTALL)
    if match is None:
        print(f"Problem {nr}: no <translation> tags, stopping. Answer was:\n{answer}")
        break

    done.append({"nr": nr, "en": english, "et": match.group(1).strip()})

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=1)
    print(f"{nr} done")