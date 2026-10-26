import json
from datasets import load_from_disk
from google import genai
import os

KEEL = "Finnish"
FAIL = "tolked_fi.json"
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

client = genai.Client()

if os.path.exists(FAIL):
    with open(FAIL, encoding="utf-8") as f:
        tulemused = json.load(f)
else:
    tulemused = []

tehtud = []
for rida in tulemused:
    tehtud.append(rida["nr"])

print("Juba tehtud:", len(tehtud))

for i in range(len(andmed)):
    if i in tehtud:
        continue
    viip = viip_mall.format(target_lang=KEEL, source_text=andmed[i]["en"])

    try:                                   
        vastus = client.interactions.create(
            model="gemini-3.6-flash",
            input=viip
        )
    except Exception as viga:             
        print(i, "VIGA:", viga)           
        continue                           

    tekst = vastus.output_text
     
    if "<translation>" in tekst and "</translation>" in tekst:
        osad = tekst.split("<translation>")
        pärast_algust = osad[1]
        osad2 = pärast_algust.split("</translation>")
        tõlge = osad2[0].strip()


        tulemused.append({"nr": i, "tolge": tõlge})
        print(i, "valmis")
        print(tõlge)
        with open(FAIL, "w", encoding="utf-8") as f:
            json.dump(tulemused, f, ensure_ascii=False, indent=2)
    else:
        print(i, "Hoiatus, silte ei leitud")

