# Keelemudelite arutluskeel eesti keeles (HTG 2026/2027 UPT)

Uurin, kuidas arutluskeel mõjutab suurte keelemudelite vastuste täpsust matemaatikaülesannete lahendamisel. Uuringuks kasutan Barua jt (2026) uuringut "Long Chain-of-Thought Reasoning Across Languages". Jälgin nende uuringut võimalikult täpselt, et saaksin enda tulemusi võrrelda. Lisan uuringule juurde eesti keele.

Juhendaja: Kristjan-Julius Laak (TÜ), Joana Jõgela (HTG)
Põhineb: Barua jt (2026), https://github.com/Berkeley-NLP/Multilingual-Long-CoT

## Failid

| Fail | Mida teeb |
|---|---|
| `harjutused/yl1.py` | andmestiku tundmaõppimine, ülesanded raskustasemete kaupa |
| `harjutused/yl2.py` | 150 ülesande valik kolmest raskusrühmast, seeme 20262609 |
| `harjutused/yl3.py` … `yl3d.py` | masintõlge Gemini 3.6 Flashiga Barua jt viiba järgi |
| `harjutused/yl4.py` | kontrolltabel 150 ülesande käsitsi kontrolliks |
| `harjutused/yl5.py` | esimene päring mudelile Ollama kaudu |
| `harjutused/yl6.py` | kolm tingimust (En-Only, En-CoT, Target-CoT) ja keele sundimine |
| `harjutused/yl7.py` | vastuste hindamine (lm-evaluation-harness, nagu Barua jt) |
| `harjutused/yl8.py` | keelekontroll (Barua jt language_validation.py, eesti keel lisatud) |
| `harjutused/yl9.py` | kogu katse ühes failis, proov 3 ülesandega |
| `valitud_150.json` | käsitsi kontrollitavate ülesannete numbrid |
| `tolked_et.json`, `tolked_fi.json` | 500 masintõlget eesti ja soome keelde |

## Kuidas käivitada

1. Klooni Barua jt repo kausta `Multilingual-Long-CoT`.
2. Tee virtuaalkeskkond ja paigalda teegid:
   `pip install datasets pandas openpyxl requests google-genai lingua-language-detector`
3. Gemini API võti peab olema keskkonnamuutujas `GEMINI_API_KEY` (mitte koodis).
4. Mudelite jaoks paigalda Ollama ja `ollama pull deepseek-r1:1.5b`.
5. Käivita failid kaustast `upt`, näiteks `python harjutused/yl9.py`.

## Muudatused Barua jt koodis

- `harjutused/language_validation.py`: lisatud eesti keel (`"Estonian": LinguaLanguage.ESTONIAN`).

## Tehisaru kasutus

Koodi kirjutamisel ja selgitamisel kasutasin Claude'i juhendamist, tõlkimiseks Gemini 3.6 Flashi. Täpsem kirjeldus on töölogis.
