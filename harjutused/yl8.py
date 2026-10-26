from language_validation import validate_language

ee = "Olgu, proovin selle välja mõelda. Meil on vaja liita kaks vektorit. Selleks liidame nende vastavad komponendid. Esimene komponent on üks pluss kaks, mis annab kolm. Teine komponent on neli pluss miinus üks, mis annab samuti kolm. Kolmas komponent on miinus kuus pluss kolm, mis annab miinus kolm."
en = "Okay, let me try to figure this out. We need to add the two vectors. To do this we add their corresponding components. The first component is one plus two, which gives three. The second component is four plus negative one, which also gives three. The third component is negative six plus three."
lyhike = "Olgu, proovin selle välja mõelda."
kordus = "Ette vahel tevik, kui ette vahel tevik. " * 60
soome = "Selvä, yritän selvittää tämän. Meidän täytyy laskea yhteen kaksi vektoria. Tätä varten laskemme yhteen niiden vastaavat komponentit. Ensimmäinen komponentti on yksi plus kaksi, mikä antaa kolme. Toinen komponentti on neljä plus miinus yksi."

testid = {"eesti": ee, "inglise": en, "lühike": lyhike, "kordus": kordus, "soome": soome}

for nimi, tekst in testid.items():
    tulemus = validate_language(tekst, "Estonian", 0.9, 0.9) # nagu Barua: 90% kindlus, 90% tükke
    print(nimi, "| vastab:", tulemus[0], "| osakaal:", round(tulemus[1], 2), "| tükke:", tulemus[2])