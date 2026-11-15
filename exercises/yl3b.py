from datasets import load_from_disk
from google import genai

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
<English>#
{source_text}
</English>""" #Translation prompt mis on samasugune barua jt omaga, et tulemusi saaks vorrelda
#target lang ja source txt on lyngad mida taidab formati abil, teeb elu kergemaks kui vahetada 2 sona ara mitte terve uus viip teha
viip = viip_mall.format(target_lang="Estonian", source_text=andmed[0]["en"]) #t2idab lyngad sihtkeel eesti keel ja valib tolgitava ylesande
client = genai.Client()

vastus = client.interactions.create( #klient suhtleb googleiga ja votab api votme 
    model="gemini-3.6-flash", #sarnaseim mudel mida nad kasutasid suthes
    input=viip #tekst mis mudelile saadetakse
)
print(vastus.output_text) #prindib terve mudeli vastuse

osad = vastus.output_text.split("<translation>")
pärast_algust = osad[1]
osad2 = pärast_algust.split("</translation>")
tõlge = osad2[0].strip()

# prindib ainult puhta tõlke
print(tõlge)