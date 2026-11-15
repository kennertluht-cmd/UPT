# Check the Fable translations:
# 1) are all 500 problems there, 2) is any translation empty,
# 3) are the formulas the same in English and Estonian.

import json
import re
from collections import Counter

with open("translated_et_fable.json", encoding="utf-8") as f:
    data = json.load(f)

print("Translations:", len(data))

numbers = {row["nr"] for row in data}
missing = [nr for nr in range(500) if nr not in numbers]
print("Missing problems:", missing)

empty = [row["nr"] for row in data if not row["et"].strip()]
print("Empty translations:", empty)

# Formulas: $...$, $$...$$, \[...\], \(...\) and [asy] drawing code
MATH = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\[asy\].*?\[/asy\]", re.DOTALL)

def formulas(text):
    found = MATH.findall(text)
    # A comma or full stop at the end of a formula may move in Estonian,
    # that is not an error, so we remove it before comparing
    cleaned = [re.sub(r"[,.;]\s*(\$+)$", r"\1", f) for f in found]
    return Counter(cleaned)

different = []
for row in data:
    if formulas(row["en"]) != formulas(row["et"]):
        different.append(row["nr"])

print("Formulas differ in", len(different), "problems:", different)
# Show what is different in each of those problems
for row in data:
    if row["nr"] in different:
        en = formulas(row["en"])
        et = formulas(row["et"])
        print("\n=== Problem", row["nr"], "===")
        print("Only in English:", list((en - et).elements()))
        print("Only in Estonian:", list((et - en).elements()))