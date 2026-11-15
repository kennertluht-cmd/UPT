tekst = "Siin on tõlge:\n<translation>\nTere maailm\n</translation>\nLoodan, et aitas!"

osad = tekst.split("<translation>")
print(osad)

pärast_algust = osad[1]
print(pärast_algust)

osad2 = pärast_algust.split("</translation>")
print(osad2)

tõlge = osad2[0].strip()
print(tõlge)