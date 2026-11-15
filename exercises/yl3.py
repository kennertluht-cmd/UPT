from google import genai #sisse google teek

client = genai.Client() # suhtleb googleiga, paneb mu api votme kylge ja saadab oigesse kohta, api voti on mul ~/.zshrc failis, et keegi ei naeks koodis mu votit.

vastus = client.interactions.create(
    model="gemini-3.6-flash", # mudel mida google soovitab, kuna see on sarnane mis barua kasutas (nende oma enam ei ole kuna liiga vana)
    input="Translate to Estonian: What is the smallest positive integer?" #tekst mis gemni saab
)

print(vastus.output_text) #ananb aind teksti mitte tokenite koguse jne jne