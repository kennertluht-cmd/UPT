import json # t66riist json faili kirjutamiseks
from datasets import load_from_disk #t66riist, mis impordib kettalt infot siia 
import random #t66riist juhuslikkuse genereerimiseks

andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500") #andmed panen muutujasse andmed

kerged = [] # kerged ülesanded list
keskmised = [] # keskmised ülesanded list
rasked = [] # rasked ülesanded list

for i in range(len(andmed)): # kõigi ülesannete läbimine
    if  andmed[i]["level"] <= 2: # kui tase on 2 voi alla
        kerged.append(i) # Lisa kergete listi ülesande number i
    elif andmed[i]["level"] == 3: #kui tase on 3 
        keskmised.append(i) # lisa listi keskmised
    else: # kui midagi muud ehk 4 voi 5
        rasked.append(i) # lisa listi rasked

random.seed(20262609) #kindel viis juhuslikult valimiseks,nii et igakord tuleb sama valik kui kasutada seda seemet.
valitud_kerged = random.sample(kerged, 50) # seemne jargi valitud suvaliselt kerged 50 ulesannet
valitud_keskmised = random.sample(keskmised, 50) #seemne jargi valitud suvaliselt keskmised 50 ulesannet
valitud_rasked = random.sample(rasked, 50) #seemne jargi valitud suvaliselt rasked 50 ulesannet
valitud = valitud_kerged + valitud_keskmised + valitud_rasked #koik valitud ulesanded kokku

print (len(valitud)) #Prindib valitud ulesannete koguarvu 

with open("valitud_150.json", "w") as f: # Avab faili nimega valitud_150.json kirjutamiseks
    json.dump(valitud, f) # Kirjutab valitud ulesannete indeksid faili valitud_150.json