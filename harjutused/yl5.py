import requests # t66riist veebipäringute saatmiseks
from datasets import load_from_disk # tööriist, mis loeb andmestiku kettalt
andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500") # andmestik muutujasse andmed

nr = 475
juhis = "Please reason step by step, and put your final answer within \\boxed{}. Your thinking and final solution must be in English."
kysimus = andmed[nr]["en"] + "\n" + juhis
paring = {
    "model": "deepseek-r1:1.5b", #mis mudel
    "prompt": kysimus   , #küsimus
    "stream": False, # oota terve vastus ära saada tükkhaaval
    "think": True, # mõttekäik eraldi välja
    "options": {
        "temperature": 0.6, # samad seaded mis Baruas
        "top_p": 0.95,
        "num_predict": 2048, # proovis lühem limiit et ei ootaks kaua
    },
}

vastus = requests.post("http://localhost:11434/api/generate", json=paring) # saada päring
tulemus = vastus.json() # uus nimi andmestik jääb alles

print("VASTUS:")
print(tulemus["response"])
print("ÕIGE VASTUS:", andmed[nr]["answer"])
print("Tokeneid:", tulemus["eval_count"])