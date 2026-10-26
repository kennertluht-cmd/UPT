from datasets import load_from_disk # tööriist, mis loeb andmestiku kettalt
import json # tööriist JSON failide lugemiseks
import pandas as pd # tööriist tabelitega töötamiseks
andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500") # andmestik muutujasse andmed

with open("valitud_150.json", encoding="utf-8") as f: # avab faili, kus on 150 valitud ülesande numbrid
    valitud = json.load(f) # loeb numbrid muutujasse valitud

with open("tolked_et.json", encoding="utf-8") as f: # avab tõlgete faili
    tolked_list = json.load(f) # loeb tõlked nimekirja tolked_list

tolked = {} # tühi sõnastik
for rida in tolked_list: # käi läbi kõik 500 rida
    tolked[rida["nr"]] = rida["tolge"] # võti on number, väärtus on tõlge

read = []                       # tühi nimekiri, siia kogume read
for nr in valitud:              # nr on järjest 475, 146, 271 ...
    uus_rida = {
        "nr": nr,
        "tase": andmed[nr]["level"],
        "en": andmed[nr]["en"],
        "mt": tolked[nr],
        "parandatud": tolked[nr],
        "vea_tyyp": "",
    }
    read.append(uus_rida)       # lisa valmis rida nimekirja

tabel = pd.DataFrame(read) # teeb 150 reast tabeli
tabel.to_excel("kontroll_150.xlsx", index=False) # salvestab tabeli Exceli faili
print("Valmis")