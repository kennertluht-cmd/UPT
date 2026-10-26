from hendrycks_utils import last_boxed_only_string, remove_boxed, is_equiv # Barua vastuste kontroll
from language_validation import validate_language # UUS: Barua keelekontroll, eesti keel lisatud
from datasets import load_from_disk
import json
import requests

andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500")

with open("tolked_et.json", encoding="utf-8") as f: # tõlked
    tolked_list = json.load(f)
tolked = {}
for rida in tolked_list:
    tolked[rida["nr"]] = rida["tolge"]

with open("valitud_150.json", encoding="utf-8") as f: # UUS: 150 valitud ülesande numbrid
    valitud = json.load(f)

# DeepSeeki vestlusvorming
ALGUS = "<｜begin▁of▁sentence｜><｜User｜>"
ASSISTENT = "<｜Assistant｜><think>\n"

# Barua jt juhis ja vastuse algus inglise keeles
EN_JUHIS = "Please reason step by step, and put your final answer within \\boxed{}. Your thinking and final solution must be in English."
EN_ALGUS = "Okay, let me try to figure this out."

# Eesti keeles, MUSTAND, Hildebrandt vaatab üle
ET_JUHIS = "Palun arutlege samm-sammult ja kirjutage lõppvastus \\boxed{} sisse. Teie mõttekäik ja lõpplahendus peavad olema eesti keeles."
ET_ALGUS = "Olgu, proovin selle välja mõelda."

def tee_viip(kysimus, juhis, vastuse_algus): # paneb viiba kokku nagu Barua eval.py
    return ALGUS + kysimus + "\n" + juhis + ASSISTENT + " " + vastuse_algus

def kysi(viip): # saadab viiba mudelile ja tagastab vastuse
    paring = {
        "model": "deepseek-r1:1.5b",
        "prompt": viip,
        "raw": True, # viip on juba valmis, ära vorminda ümber
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "num_predict": 2048,
        },
    }
    vastus = requests.post("http://localhost:11434/api/generate", json=paring)
    return vastus.json()

def kontrolli(vastuse_tekst, oige): # kas mudeli vastus on õige, nagu Barua eval.py
    kast = last_boxed_only_string(vastuse_tekst)
    if kast is None: # kasti pole, mudel ei andnud vastust
        return False
    return is_equiv(oige, remove_boxed(kast)) # võrdleb pärast normaliseerimist, nt \frac12 = \frac{1}{2}

PROOV = [valitud[0], valitud[50], valitud[100]] # UUS: üks kerge, üks keskmine, üks raske
FAIL = "tulemused_proov.jsonl" # UUS: siia lähevad tulemused
open(FAIL, "w").close() # UUS: tühjenda fail, et vanad tulemused ei seguneks

for nr in PROOV: # UUS: iga ülesanne
    viibad = {
        "En-Only":    tee_viip(andmed[nr]["en"], EN_JUHIS, EN_ALGUS), # kõik inglise keeles
        "En-CoT":     tee_viip(tolked[nr],       EN_JUHIS, EN_ALGUS), # muutub küsimuse keel
        "Target-CoT": tee_viip(tolked[nr],       ET_JUHIS, ET_ALGUS), # muutuvad juhis ja algus
    }

    for nimi, viip in viibad.items(): # iga tingimus nagu yl7-s
        tulemus = kysi(viip)
        tekst = tulemus["response"]

        if nimi == "Target-CoT": # UUS: ainult Target-CoT peab mõtlema eesti keeles
            mottekeel = "Estonian"
        else:
            mottekeel = "English"

        vastus_oige = kontrolli(tekst, andmed[nr]["answer"]) # harjutus 7
        keel_oige = validate_language(tekst, mottekeel, 0.9, 0.9)[0] == 1 # UUS: harjutus 8

        kirje = { # UUS: üks tulemus
            "nr": nr,
            "tase": andmed[nr]["level"],
            "tingimus": nimi,
            "tokeneid": tulemus["eval_count"],
            "vastus_oige": vastus_oige,
            "keel_oige": keel_oige,
            "oige": vastus_oige and keel_oige, # Barua: mõlemad peavad olema õiged
            "vastus": tekst,
        }
        with open(FAIL, "a", encoding="utf-8") as f: # UUS: "a" = lisa faili lõppu
            f.write(json.dumps(kirje, ensure_ascii=False) + "\n")

        print(nr, nimi, "| vastus:", vastus_oige, "| keel:", keel_oige, "| tokeneid:", tulemus["eval_count"])

# UUS: kokkuvõte, mitu õiget igas tingimuses
print()
for nimi in ["En-Only", "En-CoT", "Target-CoT"]:
    kokku = 0
    oigeid = 0
    with open(FAIL, encoding="utf-8") as f:
        for rea_tekst in f: # loe fail rida-realt
            r = json.loads(rea_tekst) # muuda rida tagasi sõnastikuks
            if r["tingimus"] == nimi:
                kokku += 1
                if r["oige"]:
                    oigeid += 1
    print(nimi, ":", oigeid, "/", kokku, "õiget")