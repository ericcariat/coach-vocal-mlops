# Journal des explorations

Le `CHANGELOG` raconte les runs officiels. Ce journal raconte le reste : les
essais, les diagnostics, et surtout **les impasses**. Une conclusion rétractée
vaut plus qu'un résultat lisse — elle dit comment on a appris à ne plus se tromper.

---

## 2026-08-12 — Re-contrôle Metal : ADR-002 reconfirmé, avec chiffres

`tensorflow-metal` 1.2.0 aurait pu avoir corrigé la corruption des gradients
constatée en juillet. Plutôt que de croire l'ADR sur parole, un run de
diagnostic (`v05_metal_check`, recette v03_replica à l'identique,
`use_gpu: true`, critère écrit avant le lancement) a mesuré — et l'occasion a
servi à instrumenter le **temps d'entraînement par candidat** (`fit_s` dans
`metrics.json`, MLflow et `report.md`), désormais tracé pour tous les runs.

Résultat : effondrement identique à juillet. Les 5 seeds divergent (val_loss
0.36–2.5 contre ~0.06 sur CPU, early stopping à 6 epochs partout) et, au banc
en inférence CPU, le modèle élu fait **764 FA/h contre 47** pour v03_replica.
Metal est bien ~4,7× plus rapide par epoch (3 s contre 14 s) — vitesse pour un
modèle inutilisable. Détail daté dans `ADR-002`, section « Re-contrôles ».

Au passage, la même passe d'écoute a produit les premiers **verdicts humains
sur le banc** (page « Banc streaming » : lecture audio + jugement persisté) :
sur v03_replica au seuil 0.5, 15 FA confirmées (futurs hard negatives), 1 FA
qui était en réalité une bonne détection (vérité WhisperX à corriger), 1 extrait
inexploitable, 6 FN tous confirmés.

---

## 2026-08-12 — Reprise après deux semaines · cap fixé

Décision d'orientation : **poursuivre l'amélioration de la détection dans ce
dépôt**, et non dans `coach-vocal_etape1` (figé, et où `v04_speech_neg` — la
réponse au déficit n°1 — n'existe pas).

Objectif de sortie acté : un **notebook Jupyter pour la synthèse
bloc 5 (Deep Learning)**, rédigé une fois qu'il y aura une progression à
raconter. Voir `docs/NOTEBOOK.md` pour ce que ça implique dès maintenant.

---

## 2026-07-28 — Reconstruction MLOps

Migration de `coach-vocal_etape1` vers une architecture pilotée par configuration.
Motivations : douze scripts partageant 80 % de code, `get_spectrogram()` dupliqué
six fois, aucune traçabilité rejouable des données. Voir ADR-001 et ADR-005.

Ce qui est repris tel quel : recettes de données, protocole multi-candidats,
logique du détecteur, banc streaming et sa vérité terrain. Ce qui est ajouté :
validation des configs, registre de sources et d'architectures, suivi MLflow,
audit qualité, API, interface, tests, CI, Docker.

Point de contrôle prévu : `v03_replica` doit retrouver les ordres de grandeur du
v03 historique. Sans lui, on ne saurait pas distinguer un effet de recette d'un
effet de migration.

---

## 2026-07-21 — Le banc streaming, et ce qu'il a démoli

**Premier banc : faux.** Rappel proche de zéro pour tous les modèles, 200-360
FA/h. Diagnostic : une sonde à ±5 s montrait que les modèles *voyaient* bien le
mot (probabilité 58-100 %). Le problème était la vérité terrain — les sous-titres
automatiques dérivent de ±1 s, et surtout `yt-dlp --download-sections` découpe aux
images-clés, ce qui ajoute jusqu'à 10 s au début des fichiers. Corrigé en passant
aux alignements WhisperX de `discovery.db`, les sous-titres ne servant plus qu'à
définir des zones d'incertitude.

**Deuxième banc : instructif.** Le test par clips classe mal les modèles (voir
ADR-004). Le champion réel rate encore une occurrence sur trois et déclenche ~50
fois par heure sur de la parole continue. Cause identifiée : le modèle n'a jamais
vu de parole continue à l'entraînement — uniquement des mots isolés et des
extraits d'1 s tronqués.

## 2026-07-21 — Rétractation : les 8 clips guidés

Un run officiel a donné une F1 en baisse. Hypothèse : huit clips guidés
enregistrés très bas (pic 0.06-0.11), boostés ×10, tiraient le modèle vers un
régime acoustique atypique. Trois comparaisons appariées semblaient confirmer.

**Puis on les a retirés : le résultat était encore pire.** La cause dominante
n'était pas les clips, c'était la variance d'entraînement sur CPU (±0.03-0.06 de
F1 à seed identique). Conclusion rétractée, et protocole multi-candidats adopté
(ADR-003). C'est l'épisode qui a le plus changé la méthode de travail du projet.

## 2026-07-21 — Dose de positifs synthétiques

Sweep 0 / 100 / 500 / 2000 clips TTS. Le gain **n'est pas monotone** : optimum
vers 500 (environ deux fois les positifs réels), effondrement à 2000 où le
synthétique noie le réel. Gain confirmé sur trois seeds appariés, puis au banc
streaming (rappel 65.8 → 68.4 %, FA/h 56.5 → 52.7).

## 2026-07-21 — Piper : le babil des voix de livres audio

`fr_FR-mls` (multi-locuteur, entraînée sur des livres audio) produit 1,5 à 3,5 s
de charabia sur un mot isolé — elle attend une phrase. `siwis` et `upmc` donnent
0,6-0,8 s propres. Deux parades retenues : un point final dans le texte
(« éloquence. »), qui stabilise la prosodie, et un contrôle de durée après
rognage avec régénération. `mls` a finalement été écartée.

## 2026-07-19/20 — Le modèle avait appris un débit, pas un mot

Vingt « éloquence » prononcés au tempo réel : quinze ratés. Les mêmes, ralentis
de 8-15 % : vingt sur vingt reconnus. Le modèle avait appris la vitesse
d'élocution des clips d'entraînement. Corrigé par l'augmentation de vitesse
×0.85–1.15, qui a fait passer le contrôle guidé de 6/20 à 20/20.

## 2026-07-12 — La loss qui explose

Des heures perdues à soupçonner le taux d'apprentissage, la normalisation, le
déséquilibre des classes. C'était `tensorflow-metal` qui corrompait les
gradients. Voir ADR-002. Leçon retenue : avant de mettre en cause les
hyperparamètres, mettre en cause le backend.
