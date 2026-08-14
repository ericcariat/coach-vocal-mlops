# Run `v28_resserre` — objectifs FRR/FAR atteints à l'examen

v27 (souffle + vitesse CNN + préfixes multi-vitesses) resserré : préfixes x6,
adversariaux x50. Examen final : accuracy 97.0 %, F1 0.9167, **FRR 8.81 %**
(objectif <10) et **FAR 1.72 %** (objectif <5). Banc (archive
`eloquence_20260815_005402.json`, chrono 24 s) : 92.6 % · 12.5 @0.8, 88.9 % ·
5.7 @0.9, 85.2 % · 3.4 @0.95. Vitesses (voix de référence, @0.8) : x0.85 :
64 %, normal 91 %, x1.15 : 72 %, x1.3 : 60 % — le champion : 26/79/51/34.

Au micro (2026-08-15) : déclenche encore sur « loquence » (SUFFIXE — le
miroir des préfixes, non couvert par l'entraînement) et sur
« éloquent/éloquente ». Piste v29 : négatifs-suffixes + re-pondération des
cousins enregistrés.
