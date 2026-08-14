# Run `oww_frab30np_s46` — tête openWakeWord promue (ADR-008)

Tête 64x3 entraînée sur l'extracteur speech_embedding de Google (gelé), avec
nos données réelles uniquement : positifs à contexte réel (fond MUSAN pour les
clips micro), 12 000 fenêtres de parole française en négatif (poids ×20), et
les 122 négatifs adversariaux en poids ×30.

## Banc streaming (52.7 min, 27 occurrences — archive `eloquence_20260814_033459.json`)

| Seuil | Rappel | FA/h |
|---|---|---|
| 0.8 | 92.6% | 3.4 |
| 0.9 | 92.6% | 3.4 |
| 0.95 | 88.9% | 2.3 |
| 0.98 | 81.5% | 2.3 |
| 0.99 | 77.8% | 1.1 |

Mots proches (« éloquente », « élégance »…) : 0 % de déclenchement dès 0.8
(baseline avant correction : 16 %) — preuve `artifacts/reports/oww_cousins.png`.
Champion précédent (CNN v17, seuil 0.8) : 48.1 % · 6.8 FA/h.
Test au micro : validé au seuil 0.8.
