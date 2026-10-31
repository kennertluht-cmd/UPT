# Harjutus 3d: tõlgib 500 ülesannet Geminiga Barua jt viiba järgi, jätkab pärast katkestust
import json
from datasets import load_from_disk
from google import genai #googlei teek, python suhtleb googleiga
import os #failisüsteemi jaoks, kontrollib, kas tulemuste fail on juba olemas.

KEEL = "Finnish" #keele nimi, mis läheb viipa
FAIL = "tolked_fi.json" #faili nimi, sinna tõlked salvestatakse
andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500") 
viip_mall = """Your job is to return a translated version of the English text. 
* Translate to {target_lang}.
* The translation must be fluent, easy to read by native speakers.
* Do not solve the prompt translate it.
* You must preserve all details including math notations (latex), code, and special tokens (e.g., <think>).
* The math notations, code, and special tokens must not be translated, keep it as is.
* Return your translation in the following format.
<translation>
...translated text...
</translation>
The following is the source text for your task:
<English>
{source_text}
</English>""" 
#barua jt viibe inglisekeelsel ja muutmata kujul (barua repo failist)
#target lang ja source txt on lyngad mida taidab formati abil, teeb elu kergemaks kui vahetada 2 sona ara mitte terve uus viip teha

client = genai.Client() #loeb mu API võtme, võti on ~/.zshrc failis mitte koodis

if os.path.exists(FAIL): #Kas fail on juba olemas, kui ei siis alusta algusest
    with open(FAIL, encoding="utf-8") as f:
        tulemused = json.load(f)
else:
    tulemused = []

tehtud = []
for rida in tulemused:
    tehtud.append(rida["nr"])

print("Juba tehtud:", len(tehtud))  #kogu valmis tõlgete numbrid, et neid uuesti mitte teha

for i in range(len(andmed)): #käi läbi kõik 
    if i in tehtud: #kui on see juba tehtud
        continue #mine jargmise ulesande juurde
    viip = viip_mall.format(target_lang=KEEL, source_text=andmed[i]["en"]) #täida malli lüngad, tulemus on viip geminile

    try:                                   
        vastus = client.interactions.create(
            model="gemini-3.6-flash",
            input=viip
        )# saada viip Geminile
    except Exception as viga:         
        print(i, "VIGA:", viga)           
        continue          #kui midagi juhtub, nt internet kaob, siis ei jookse kokku vaid prindib vea ja läheb järgmise juurde                  

    tekst = vastus.output_text #gemini vastusest ainult tekst
     
    if "<translation>" in tekst and "</translation>" in tekst: #kas geimini tegi nii nagu viip palus
        osad = tekst.split("<translation>")
        pärast_algust = osad[1] 
        osad2 = pärast_algust.split("</translation>")
        tõlge = osad2[0].strip()
        #lõika tekst esimese sildi juurest, võta teine pool, lõika teise sildi juurest ja võta esimene pool. .strip() eemaldab tühikud ja reavahetused algusest ja lõpust.


        tulemused.append({"nr": i, "tolge": tõlge}) #lisa tulemuste listi uus rida koos numbri ja tolkega
        print(i, "valmis")
        print(tõlge)
        with open(FAIL, "w", encoding="utf-8") as f: #salvesta fail kohe peale igat tehtud tõlget
            json.dump(tulemused, f, ensure_ascii=False, indent=2) #täpitähed ja inimesele loetav (taandega=)
    else:
        print(i, "Hoiatus, silte ei leitud") #kui ei ole silte, siis jääb yk tegemata

