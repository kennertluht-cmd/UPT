from datasets import load_from_disk
import json
import requests
andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500")

with open("tolked_et.json", encoding="utf-8") as f: # nüüd on tõlkeid vaja
    tolked_list = json.load(f)
tolked = {}
for rida in tolked_list:
    tolked[rida["nr"]] = rida["tolge"]

# DeepSeeki vestlusvorming  hugging faceist, sama mis barua
ALGUS = "<｜begin▁of▁sentence｜><｜User｜>" 
ASSISTENT = "<｜Assistant｜><think>\n"

# Barua jt juhis ja vastuse algus inglise keeles (sundimine)
EN_JUHIS = "Please reason step by step, and put your final answer within \\boxed{}. Your thinking and final solution must be in English."
EN_ALGUS = "Okay, let me try to figure this out."

# Eesti keeles, MUSTAND, peab kellegagi kontrollima, Joana soovitas Hildebrandti üle vaadata
ET_JUHIS = "Palun arutlege samm-sammult ja kirjutage lõppvastus \\boxed{} sisse. Teie mõttekäik ja lõpplahendus peavad olema eesti keeles."
ET_ALGUS = "Olgu, proovin selle välja mõelda."

def tee_viip(kysimus, juhis, vastuse_algus): # paneb viiba kokku nagu Barua eval.py
    return ALGUS + kysimus + "\n" + juhis + ASSISTENT + " " + vastuse_algus

nr = 475 #suvaline number 
en_only = tee_viip(andmed[nr]["en"], EN_JUHIS, EN_ALGUS) #küsimus ja juhis inglise keeles 
print(en_only)
print("-----")

en_cot = tee_viip(tolked[nr], EN_JUHIS, EN_ALGUS) #muutub kysimuse keel
print(en_cot)
print("-----")

target_cot = tee_viip(tolked[nr], ET_JUHIS, ET_ALGUS) #muutub mõtlemise keel
print(target_cot)

def kysi(viip): # saadab viiba mudelile ja tagastab vastuse
    paring = {
        "model": "deepseek-r1:1.5b",
        "prompt": viip,
        "raw": True, # viip on juba valmis ära vorminda ümber
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "num_predict": 2048,
        },
    }
    vastus = requests.post("http://localhost:11434/api/generate", json=paring)
    return vastus.json()

viibad = {"En-Only": en_only, "En-CoT": en_cot, "Target-CoT": target_cot}

for nimi, viip in viibad.items(): # käi läbi kolm tingimust
    tulemus = kysi(viip)
    print("=====", nimi, "=====")
    print(tulemus["response"][-400:]) # vastuse viimased 400 märki, sp, et päris vastus pikkpikk
    print("Tokeneid:", tulemus["eval_count"])
    print("ÕIGE:", andmed[nr]["answer"])