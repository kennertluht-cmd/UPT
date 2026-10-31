#Harj. 1: Barua jt andmestiku tundmaõppimine, mitu ülesannet, mis veerud, mitu igal raskustasemel
from collections import Counter #t66riist mis loeb kokku mitu korda midagi esineb
from datasets import load_from_disk #t66riist, mis impordib kettalt infot siia 

andmed = load_from_disk("Multilingual-Long-CoT/eval/benchmarks/mt-math-500") #andmed panewn muutujasse andmed

print(len(andmed)) #annab ülesannete arvu
print(andmed) #prindib muutuja andmed, naitab. mis tabelis on mitte tabeli sisu 
print(andmed[0]["en"]) #prindib 1 kysimuse inglise keeles
print(andmed[0]["answer"]) # prindib esimese kysimuse vastuse 
print(Counter(andmed["level"])) # mitu ylesannet on igal tasemel kokku on 5 taset.