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
        "temperature": 0.6, #kui palju mudel varieerub enda vastustes. .6 on mõõdukas ja sp kulub erinev arv tokeneid iga kord
        "top_p": 0.95, #kokku peavad olema 95% tõenäosusega kõige tõenäolisemad sõnad, mida mudel võib järgmisena öelda
        "num_predict": 2048, #proovis lühem limiit et ei ootaks kaua
    }, # neid kasutasid Barua jt.
}

vastus = requests.post("http://localhost:11434/api/generate", json=paring) # saada päring ollamale
tulemus = vastus.json() # muuda vastus sõnastikuks

print("VASTUS:") #prindib sõna vastus
print(tulemus["response"]) #mudeli kirjutatud vastus
print("ÕIGE VASTUS:", andmed[nr]["answer"]) #prindib õige vastuse andmestikust
print("Tokeneid:", tulemus["eval_count"]) #mitu tokenit mudel kasutas vastuseks.