# Run `oww_frab30np4b_s46` — le modèle qui attend la fin du mot

Recette du champion précédent (contexte réel, fond MUSAN, français ×20,
adversariaux ×30) plus deux correctifs de la fuite « tire avant la fin » :
négatifs-préfixes 60/75/85 % du mot (poids ×8) et pool adversarial porté à
145 clips (23 enregistrements micro « éloquente / éloquen » du 2026-08-14,
dont 10 FP du champion précédent à proba 0.98-0.9999).

## Banc streaming (52.7 min, 27 occurrences — archive `eloquence_20260814_211029.json`)

| Seuil | Rappel | FA/h |
|---|---|---|
| 0.8 | 81.5% | 2.3 |
| 0.9 | 81.5% | 2.3 |
| 0.95 | 77.8% | 1.1 |

Préfixes tronqués 80/90 % du mot : 0 % (champion précédent : 17 %/38 %).
Cousins et « éloquente » : 0 %. Test micro : validé — « éloquence » détecté,
« éloquen » et « éloquente » muets (verdict du 2026-08-14).
Champion précédent (oww_frab30np_s46, seuil 0.8) : 92.6 % · 3.4 FA/h — le
recul de rappel YouTube (22/27 contre 25/27) est le prix accepté de la
fiabilité sur la voix cible (priorité déclarée).
