# Run `v27_prefrapides` — l'oreille record, pas encore la retenue

Souffle partout + augmentation vitesse du CNN + négatifs-préfixes découpés
aussi en versions accélérées (le conflit durée/fin-du-mot désamorcé, mesuré).

Banc (52,7 min / 27 occ, archive `eloquence_20260815_004536.json`) :
96.3 % (26/27) · 41.0 FA/h @0.8 · 92.6 % · 15.9 @0.95. Vitesses (voix de
référence) : lent x0.85 : 94 %, normal : 100 %, x1.15 : 68 %, x1.3 : 40 %.
Examen : FRR 9.84 % (objectif <10 atteint) mais FAR 7.22 % (objectif <5
manqué). Diagnostic : le rééquilibrage a désserré tous les négatifs.
NON promouvable — v28 resserre (préfixes x6, adversariaux x50).
