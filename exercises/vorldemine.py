import json
from datasets import load_from_disk

andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500")

with open("tolked_proov.json", encoding="utf-8") as f:
    tolked = json.load(f)

for rida in tolked:
    nr = rida["nr"]
    print("NR:", nr)
    print("EN:", andmed[nr]["en"])
    print("ET:", rida["et"])
    print()