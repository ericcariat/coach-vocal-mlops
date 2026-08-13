# Historique des runs et des promotions

Un run = un dossier `artifacts/runs/<mot>/<id>/` (config.json, manifest.csv,
metrics.json, figures, report.md, model.keras, candidates/). Ce fichier dit
**quelle stratégie** a changé et **pourquoi** un modèle a été promu ; le détail
chiffré vit dans chaque `report.md`.

Le modèle en production est décrit par `artifacts/runs/<mot>/CHAMPION.json` et le
lien `current/`. Ne jamais écrire un chemin de run en dur ailleurs.

---

## Convention

- Nouvelle stratégie = nouvelle **configuration** (`configs/experiment/*.yaml`),
  pas un nouveau script.
- Après le banc streaming, si le run bat le champion : `coachvocal registry
  promote <run> --reason "<chiffres>"`, puis une entrée ici.
- Métriques suivies : **rappel streaming** et **FA/heure** (décision) ;
  **F1, FRR, FAR, AUC** par clip (contrôle).

---

## Runs

### v10_recut / v10b (1ᵉ version) — 2026-08-13 — ❌ un bug de découpe, pas la géométrie
- **Symptôme** : rappel « 100 % » mais 782-810 FA/h — le modèle tirait sur tout.
- **Premier diagnostic (faux)** : « la géométrie seule casse tout ». La
  vérification par corrélation croisée a montré la vraie cause : **~2/3 des
  fenêtres re-découpées ne contenaient pas le mot**. Deux pièges superposés :
  `t_end` de discovery.db n'est pas la fin du mot (spans jusqu'à 15 s), et les
  temps dérivent de −0.1 à −0.9 s selon les segments (ré-encodage yt-dlp).
  Les « positifs » étaient donc en grande partie de la parole quelconque —
  d'où « tout est positif ».
- **Correctif** : la source `word_clips_recut` localise désormais chaque mot
  **par corrélation croisée avec le clip propre d'origine** (précis à
  l'échantillon, auto-vérifiant : occurrence introuvable → écartée, jamais
  découpée à l'aveugle), et mesure la longueur du mot sur le padding de zéros
  du clip. Vérifié : 24/25 fenêtres échantillonnées contiennent le mot, début
  entre 0.03 et 0.38 s (fin près du bord). Leçon pour le notebook : une
  vérité terrain décalée ne donne pas un résultat imprécis, elle fabrique un
  résultat absurde — deuxième occurrence du motif (cf. banc de juillet).

### Piste B, round 3 (ocean_ctx) — 2026-08-13 — l'hypothèse du contexte CONFIRMÉE
- **Protocole** : positifs à CONTEXTE RÉEL (1 862/1 968 fenêtres de 2 s
  relocalisées par corrélation croisée dans les segments, fin du mot au bord ;
  106 replis padding) + océan ACAV 11 h. 3 seeds.
- **Résultat** : le rappel revient — **92-100 %**, avec « nu » 2/2 et « d' »
  8-9/9 SYSTÉMATIQUES (ces têtes détectent tout ce que le CNN rate). Mais la
  frontière reste lâche : 235-645 FA/h.
- **Lecture** : le remède contexte fonctionne (hypothèse du round 2 confirmée
  par l'expérience) ; en contrepartie les positifs ressemblent désormais aux
  négatifs de flux — 11 h d'océan ne suffisent plus à séparer (eux : 2 000 à
  31 000 h). **Round 4 en attente** : le fichier ACAV 2 000 h (16 Go, en
  téléchargement) — le dernier ingrédient d'échelle.

### Piste B, round 2 (ocean11h) — 2026-08-13 — la balançoire, et sa cause
- **Protocole** : mêmes positifs réels + 60 167 fenêtres de négatifs ACAV
  pré-calculés (~11 h, ratio 30:1). 3 seeds.
- **Résultat** : la balançoire re-bascule — rappel 11-59 % à 0.2 (0-15 % à
  0.8) pour 2-15 FA/h. Retour au profil « synthétique ».
- **Cause identifiée (testable)** : nos positifs d'entraînement sont padés de
  SILENCE avant le mot (clips 1 s complétés à 2 s), le banc présente le mot
  après de la VRAIE parole. Avec peu de négatifs, la tête généralisait ;
  noyée de flux continus réels étiquetés négatifs, elle a appris « contexte
  continu = négatif ». C'est la leçon v10 (padding-signature) rejouée dans
  l'espace des embeddings.
- **Remède désigné** : des positifs oWW à CONTEXTE RÉEL — fenêtres de 2 s
  découpées dans les segments du corpus, fin du mot au bord, par la
  corrélation croisée de `word_clips_recut` (l'outil existe déjà). Les deux
  chantiers de la journée (recut + greffe) convergent exactement ici.

### Piste B (têtes oWW sur nos données) — 2026-08-13 — la décomposition prouvée
- **Protocole** : tête 64x3 entraînée EN LOCAL (scripts/train_oww_head.py) sur
  le front-end gelé oWW : 1 968 positifs réels + 6 921 négatifs de nos
  recettes (~2,6 h). 3 seeds, export ONNX vérifié bit-à-bit, banc habituel.
- **Résultat, miroir exact des têtes synthétiques** : rappel **92-100 %**
  (« nu » 2/2 — les seules détections jamais mesurées ! — et « d' » 9/9)
  mais **500-1 200 FA/h**. Les synthétiques : 16-24 % · 1.1 FA/h.
- **Verdict — la décomposition est démontrée dans les deux sens** : la
  représentation pré-entraînée transfère parfaitement (nos voix réelles y
  dessinent la BONNE région) ; le silence, lui, vient exclusivement du volume
  de négatifs (~31 000 h chez eux vs nos 2,6 h : la frontière n'est pas
  sculptée). **Le quadrant gagnant = nos positifs réels + LEUR océan de
  négatifs** — leurs features de négatifs pré-calculées sont téléchargeables
  (format [16×96] directement compatible avec notre script). Prochain geste.

### v17_stack — 2026-08-13 — ⭐ PROMU (seuil live 0.8, priorité silence)
- **Décision de l'auteur** : priorité aux FA/h. Au banc courant (53,1 min, 27 occ.,
  2 nus difficiles inclus) : **48.1 % · 6.8 FA/h** au seuil 0.8 — contre
  63.0 % · 29.4 pour v11 : **FA ÷ 4.3**, au prix du rappel banc assumé.
- Mesure décisive du balayage : **0.7 est un point DOMINÉ** (9.1 FA/h comme
  0.65 mais −3.7 pts) — seuls 0.65 (55.6 · 9.1) et 0.8 (48.1 · 6.8) sont des
  choix rationnels ; 0.65 reste la réserve « rappel » (une ligne de config).
- Contexte personnel : v17 fait **10/10 (proba 1.00)** sur la voix de l'auteur au
  mot nu en test guidé — le rappel banc mesure des voix YouTube/réunion, pas
  l'usage micro réel.
- Seuil live : déjà 0.80 dans la config du mot — aucun changement nécessaire.

### v18_guided — 2026-08-13 — ❌ la boucle personnelle dégrade le banc
- **But** : v17 + les 33 essais guidés de l'auteur (10 TP mot nu ×10, 15 FP
  « éloquente/élégance » + 8 TN ×5). Critère : ≥ v17 − 2 pts et FA/h ≤ v17.
- **Banc (0.55)** : 51.9 % · 39.8 FA/h contre 63.0 % · 18.2 pour v17 — ❌ sur
  les deux axes.
- **Verdict** : surpondérer une acoustique unique (une voix, un micro, une
  pièce — 310 positifs effectifs) fait glisser le modèle hors du domaine du
  banc. **Enseignement de fond : l'objectif « banc » (voix variées, champ
  lointain) et l'objectif « ma voix au micro » sont deux cibles distinctes qui
  peuvent diverger.** La voie propre pour la seconde : boost modeste (×2-3),
  ou un banc personnel dédié, ou une adaptation finale légère — pas un ×10.
  Side-note : la session guidée elle-même a mesuré v17 à 10/10 sur la voix
  de l'auteur (mot nu, proba 1.00) — le besoin personnel est déjà couvert par v17.

### v17_stack — 2026-08-13 — la courbe domine, le point de fonctionnement bouge
- **Recette** : cumul rappel-d'abord — RIR/multi-SNR (+ rappel), dose 300
  YouTube, SUMM-RE 75, hard negatives ×2, élection fa_ambient (élu : seed 42,
  1.74 FA/h ambiantes).
- **Critère pré-déclaré (seuil 0.8)** : ❌ au pied de la lettre (51.9 % à 0.8 —
  la recette décale la calibration vers le bas). MAIS le balayage de seuils
  montre que **la courbe de v17 domine celle du champion sur toute sa longueur
  utile** : à 0.55 → **70.4 % · 18.1 FA/h** (champion@0.8 : 63.0 · 29.4 —
  mieux sur les DEUX axes) ; à 0.65 → 63.0 % · **9.0 FA/h** (rappel égal,
  FA ÷ 3.3). Et « d'éloquence » : **8/10** (record, effet RIR).
- **Honnêteté méthodologique** : promouvoir v17 implique de changer AUSSI le
  seuil de fonctionnement live (0.8 → 0.55 ou 0.65), un paramètre produit —
  le critère écrit ne l'avait pas prévu. Décision laissée à l'auteur : point
  « rappel » (0.55) ou point « silence » (0.65). Pas de promotion automatique.

### Banc recomposé + v15b/v16 — 2026-08-13 — le champion tient, la frontière est claire
- **Banc recomposé** : les 10 vidéos sacrifiées aux hard negatives sont
  sorties, remplacées automatiquement — 53,1 min, **27 occurrences dont 10
  « d'éloquence »** (vs 7 avant : plus dur sur la forme faible). Nouvelle
  référence champion : **63.0 % · 29.4 FA/h** @0.8 (les 76 % d'avant
  mesuraient un autre échantillon — ne jamais comparer entre bancs).
- **v15b_summre_75** : 55.6 % · 15.8 — FA −46 % mais rappel −7.4 pts ❌.
- **v15b_summre_150** : 40.7 % · 26.0 ❌.
- **v16_hardneg (boost ×10)** : 33.3 % · **4.5 FA/h** — le plus silencieux
  jamais mesuré (0.0 FA/h ambiantes !) et le plus sourd. Le boost ×10
  (540 fenêtres effectives pour 18 FA réelles) était une overdose — à
  réessayer à ×2-3.
- **Le motif de la journée, énoncé** : TOUT levier anti-FA basé sur des
  négatifs paie en rappel — le modèle est à sa frontière de capacité. Le seul
  levier qui a MONTÉ le rappel est v12 (RIR, +12 pts). La suite logique n'est
  plus « encore des négatifs » mais le CUMUL rappel d'abord :
  v17 = RIR + dose SUMM-RE 75 + hard negatives ×2, élu fa_ambient — ou
  accepter le point produit actuel. Champion inchangé : v11_speech_300.

### v15_summre_neg — 2026-08-13 — ❌ rappel, ✅ hypothèse de domaine
- **But** : champion + 300 négatifs de réunions SUMM-RE (split train, réunions
  et locuteurs disjoints du banc/val_ambient — test anti-fuite dédié) +
  élection fa_ambient. Critère : FA/h ≤ 27 ET rappel ≥ 74 %.
- **Banc étendu (seuil 0.8)** : **8.7 FA/h** (✅, dont réunions 17 → 2 FA —
  l'hypothèse « entraîner sur le domaine qui fait mal » est confirmée net)
  mais rappel **52.0 %** (❌, −24 pts). Effet secondaire notable : la recette
  écrase aussi la dispersion ambiante des seeds (5.2-7.0 FA/h contre 8.7-31.4
  sans — l'élection n'avait presque plus rien à trier).
- **Verdict** : motif v04 rejoué — 600 fenêtres continues au total assourdissent.
  Chaque domaine de négatifs a SA dose : le sweep SUMM-RE (75/150) reste à
  faire, et les hard negatives ciblés (FA confirmées à l'oreille) restent la
  voie sans coût de rappel. Champion inchangé (v11_speech_300).

### Comparatif openWakeWord — 2026-08-13 — le banc tranche : 5× moins de rappel
- **Protocole** : 5 têtes oWW « éloquence » (64x3, 115 k steps, entraînées par
  l'auteur sur le pipeline openWakeWord, ~100 % synthétique) passées sur NOTRE banc
  étendu via l'adaptateur ONNX (leur front-end mel+embeddings, NOTRE machine à
  états), seuils 0.05 → 0.8.
- **Résultats (seuil 0.5 ; les 5 modèles se valent)** : rappel **16-24 %** ·
  **1.1 FA/h** — contre **76 % · 33.9** pour le champion v11_speech_300. Même à
  0.05, le meilleur plafonne à 36 %. Et **« d'éloquence » : 0/7 pour les cinq
  modèles**, à tous les seuils — Piper ne produit pas cette élision, le modèle
  ne l'a jamais entendue (prédiction faite avant le run, confirmée).
- **Lecture honnête** : (1) leur « 50 % de rappel » interne était mesuré sur
  clips (large part synthétique) — sur du VRAI français en flux, il fond à
  ~20 % : l'entraînement tout-synthétique ne généralise pas aux voix réelles ;
  (2) leur FA/h ~1 est remarquable — le transfer learning sur embeddings est
  très silencieux ; (3) réserve méthodologique : notre règle des 3 fenêtres à
  leur cadence 80 ms (240 ms de persistance) peut coûter un peu de rappel,
  mais pas un facteur 3-5. Verdict : notre CNN sur données réelles reste
  largement devant en rappel ; l'approche oWW garde l'atout FA/h et la taille
  (405 Ko). Preuves : `artifacts/reports/stream_bench/eloquence.json`.

### v14_rir_speech300 — 2026-08-13 — ❌ de peu : le cumul est partiel
- **But** : cumuler les deux gains orthogonaux (dose 300 + RIR/multi-SNR).
  Critère : rappel ≥ 80 % ET FA/h ≤ 45.
- **Banc étendu (seuil 0.8)** : rappel **84.0 %** (21/25, d' 4/7) ·
  **51.4 FA/h**. ✅ rappel, ❌ FA/h (51.4 > 45). Vs champion v11 : +8 pts de
  rappel pour +17.5 FA/h — un compromis, pas une domination.
- **Verdict** : pas de promotion (critère non atteint, et le champion garde
  les FA/h). Le cumul retient l'essentiel du gain de rappel de v12 mais pas
  toute la baisse de FA de v11. **Suite la plus prometteuse** : v14 élu par
  `fa_ambient` (v13 a montré ×3,6 de dispersion entre seeds) — le candidat
  le plus silencieux de cette recette pourrait passer sous 45.

### v13_fa_select — 2026-08-13 — ✅ critère atteint : l'élection vaut une recette
- **But** : élire les candidats par FA/h sur 34,5 min de flux ambiant SUMM-RE
  (hors banc) sous contrainte rappel val ≥ 90 %, au lieu de min(val_loss) —
  recette v03 par ailleurs inchangée.
- **Dispersion mesurée** : à recette identique, les 5 seeds vont de **8.7 à
  31.4 FA/h ambiantes** (×3,6) — c'est ce que val_loss ne voit pas.
- **Banc étendu (seuil 0.8)** : 76.0 % · **31.7 FA/h** — critère ✅ (< 66.7 à
  rappel ≥ 74 %). Égal au champion v11 (33.9) à 2 FA près = bruit (ADR-003) :
  **pas de re-promotion**, mais l'élection produit seule retrouve le gain de
  la dose 300. Les deux mécanismes devraient se cumuler (backlog : v14 +
  élection fa_ambient).

### v12_rir — 2026-08-13 — ❌ au critère strict, mais le meilleur rappel mesuré
- **But** : combler l'écart d'augmentation avec l'état de l'art (RIR MIT +
  bruit multi-SNR 5-20 dB, p=0.5 chacun, recette v03 sinon inchangée).
- **Banc étendu (seuil 0.8)** : rappel **88.0 %** (22/25 — record, +12 pts) et
  « d'éloquence » remonte de 3/7 à **5/7** ; mais **72.2 FA/h** > critère
  (≤ 66.7) et surtout > champion v11 (33.9). ❌ strict, gain de rappel réel —
  la réverbération attaque bien le déficit champ-lointain vu sur SUMM-RE.
- **Suite** : v14_rir_speech300 teste le cumul des deux gains orthogonaux
  (dose 300 + RIR), critère : rappel ≥ 80 % ET FA/h ≤ 45.

### v11_speech_300 — 2026-08-13 — ⭐ PROMU : la dose utile de parole continue
- **Sweep dose-réponse 0/100/300/500** (train seulement, val/test intacts —
  contrairement à v04), critère écrit avant les runs : minimiser les FA/h au
  banc sous contrainte rappel ≥ référence − 2 pts.
- **Banc étendu (seuil 0.8)** : dose 0 → 76.0 % · 66.7 FA/h ; dose 100 →
  76.0 % · 91.9 ; **dose 300 → 76.0 % · 33.9 (−49 %)** ; dose 500 → 68.0 % ·
  33.9 (le rappel commence à céder). Clips : F1 0.9347 (comparable, val/test
  inchangés).
- **Verdict** : la courbe complète confirme le récit v04 : 1500 était
  l'overdose, 500 en est le début, 300 est l'optimum — FA/h divisées par deux
  à rappel strictement identique. **Promu champion** sur les faits du banc.
- Reste vrai : « d'éloquence » 3/7 — la dose ne corrige pas la forme faible
  (piste données ciblée, backlog).

### v10b_recut_speech (découpe corrigée) — 2026-08-13 — ❌ sous la référence
- **Banc étendu (seuil 0.8)** : rappel 64.0 % (16/25) · 118.1 FA/h — contre
  76.0 % · 66.7 pour v03_replica. Critère ❌ sur les deux volets. À 0.5 :
  80 % · 378/h.
- **Verdict** : même correctement découpée, la géométrie fin-de-fenêtre (+300
  négatifs continus) ne bat pas l'ancienne sur ce banc, à architecture et
  protocole constants. **Hypothèse mécanique à tester avant d'enterrer la
  piste** : la règle de décision exige 3 fenêtres consécutives > seuil, or un
  modèle entraîné « fin de mot au bord » ne pique par construction que sur
  ~1-2 positions de fenêtre (hop 125 ms, jitter 200 ms) — la géométrie l'a
  rendu PRÉCIS en temps, et la machine à états punit cette précision.
  À explorer : jitter élargi (~400 ms) ou n_consecutive adapté — mais c'est
  une expérience à part entière (la machine à états est commune banc/live,
  la changer re-calibre tout). La géométrie historique reste en place.

### Banc étendu — 2026-08-13 — nouvelle référence de mesure
- **54,9 min** (18 segments YouTube + 3 pistes SUMM-RE de 42,4 min), 25
  occurrences. Le banc reporte désormais le **rappel par forme** (nu/l'/d').
- **v03_replica sur ce banc (seuil 0.8)** : rappel 76.0 % (19/25) ·
  **66.7 FA/h** — la nouvelle référence pour toutes les comparaisons.
- **Deux découvertes** : (1) les réunions françaises ordinaires déclenchent
  PLUS de FA que le corpus thématique (51 des 61 FA sur SUMM-RE, ~72/h contre
  ~48/h sur YouTube) — l'hypothèse « thématique = pire cas » est réfutée, la
  robustesse au champ lointain devient prioritaire (→ v12_rir) ; (2) le rappel
  s'effondre sur la forme élidée **« d'éloquence » : 3/7 (43 %)** contre 13/15
  (87 %) pour « l'éloquence » — piste de données ciblée.

### v09_gate — 2026-08-13 — 🔬 porte qualité : FA/h −39 % mais rappel −8 pts @0.8
- **But** : valider la porte ADR-007 de bout en bout (recette v03 filtrée :
  134 rejetés + 622 douteux exclus en attendant l'audit humain).
- **Banc étendu (seuil 0.8)** : rappel 68.0 % (17/25) · **40.5 FA/h** — contre
  76.0 % · 66.7 pour v03_replica. Critère ❌ sur le rappel (seuil : ≥ 74 %),
  ✅ sur les FA/h. À 0.5, v09 DOMINE (76.0 % · 85.3/h contre 72.0 % · 118.1) —
  le compromis dépend du seuil.
- **Verdict** : l'infrastructure de la porte est validée ; sa SÉVÉRITÉ actuelle
  (622 douteux exclus dont 336 positifs) coûte du rappel à seuil haut.
  Prochain geste : l'audit humain des douteux (page Qualité) réintégrera les
  vrais positifs — puis re-run. Pas de promotion.

### v08_metal_maxrelu — 2026-08-13 — ❌ le test d'attribution condamne le backend
- **But** : séparer l'effet backend de l'effet activation, ce que v06/v07 ne
  permettaient pas. `relu_max` (`tf.maximum(x, 0)`) est mathématiquement
  identique au ReLU : recette v03 à l'identique fonctionnel, seul le backend
  change. Rouvre en conscience la série close par v07 (assumé dans la config).
- **Entraînement impeccable — sur le papier** : val_loss 0.056–0.10, au niveau
  du CPU, F1 clips 0.9322 (élu : seed 42), ~4,7 s/epoch (5 min 50 s le run, ×3).
- **Banc (inférence CPU, seuil 0.8)** : rappel 78.3 % mais **120.3 FA/h contre
  47.1** — 2,5× la référence, le pire des trois contournements. ❌ net.
- **Verdict — le plus instructif de la série** : à fonction mathématiquement
  identique, Metal produit un modèle dont les métriques d'entraînement et de
  clips sont indiscernables du CPU, mais qui s'effondre en FA/h au banc. Le
  backend fabrique bien des modèles subtilement différents, d'une façon
  qu'aucune métrique amont ne détecte (ADR-002 **et** ADR-004 confirmés d'un
  même geste). Dossier Metal refermé définitivement pour ce projet.
  Preuves : `artifacts/runs/eloquence/v08_metal_maxrelu/`.

### v07_metal_elu — 2026-08-12 — ❌ critère non atteint : fin de la série Metal
- **But** : deuxième essai du contournement (elu), même critère que v06.
- **Entraînement sain** : val_loss 0.08–0.10 (les plus proches du CPU),
  F1 clips 0.9286 (élu : seed 42). Mais elu est lent sur Metal : ~8,4 s/epoch,
  8 min 24 s le run — ×2 vs CPU seulement, loin du ×6 de leaky_relu.
- **Banc (inférence CPU, seuil 0.8)** : rappel **78.3 %** (18/23, une occurrence
  de mieux que v03_replica) mais **73.3 FA/h** contre 47.1 (14 FA contre 9) —
  ❌ sur le volet FA/h du critère.
- **Verdict** : clause d'arrêt pré-déclarée appliquée — deux activations
  saines (leaky, elu), deux échecs au banc : la piste Metal s'arrête, ADR-002
  inchangé, retour au travail sur les données. Nuance honnête : à cette taille
  de banc (23 occurrences, ~15 FA), une partie de ces écarts est du bruit —
  l'extension du banc (backlog) rendrait ces comparaisons plus tranchantes.
  Preuves : `artifacts/runs/eloquence/v07_metal_elu/`.

### v06_metal_leaky — 2026-08-12 — ❌ critère non atteint : Metal reste écarté
- **But** : le bug ReLU étant circonscrit aux Dense fusionnées et contourné par
  `leaky_relu`, la recette v03 entraînée sur Metal retrouve-t-elle la qualité
  CPU, et en combien de temps ? Critère pré-déclaré : rappel banc à ±2 points
  de v03_replica, FA/h du même ordre, temps nettement meilleur.
- **Entraînement sain, vitesse confirmée** : val_loss 0.12–0.16, early stopping
  normal (7–19 epochs), **2 min 39 s pour 5 candidats contre ~17 min sur CPU**
  (~3 s/epoch, ×4,7). Clips : F1 0.9120 (élu : seed 46).
- **Banc (inférence CPU, seuil 0.8)** : rappel **60.9 %** (14/23) et
  **83.7 FA/h** — contre 73.9 % · 47.1 pour v03_replica. Dégradé sur les deux
  axes : ❌ sur le critère qualité.
- **Verdict** : la vitesse ne rachète pas la qualité ; Metal reste écarté,
  ADR-002 inchangé. Attribution impossible entre backend Metal, passage à
  leaky_relu et variance (deux facteurs changés à la fois — assumé dans la
  config) : un jumeau CPU en leaky_relu trancherait si la question redevient
  d'actualité. Preuves : `artifacts/runs/eloquence/v06_metal_leaky/`.

### v05_metal_check — 2026-08-12 — 🔬 diagnostic : ADR-002 reconfirmé
- **But** : re-contrôler si `tensorflow-metal` 1.2.0 corrompt toujours
  l'entraînement, et mesurer le temps (recette v03_replica, `use_gpu: true`).
  Jamais candidat à la promotion, quel que soit le résultat.
- **Verdict** : corruption identique à juillet — 5 seeds effondrés (val_loss
  0.36–2.5), et au banc en inférence CPU : **764 FA/h contre 47** pour
  v03_replica. Metal est ~4,7× plus rapide par epoch, pour un modèle
  inutilisable. Détail : `docs/JOURNAL.md` et `ADR-002` § Re-contrôles.

### v04_speech_neg — 2026-08-12 — ❌ critère non atteint, pas de promotion
- **Hypothèse** : les négatifs de parole continue font baisser les FA/heure.
- **Critère défini à l'avance** : FA/h −20 % minimum, rappel −2 points maximum.
- **Recette** : tts500 + 1500 fenêtres de parole continue YouTube en négatifs
  (train), 150 en val/test — le test par clips n'est donc **plus comparable**
  à v03 (F1 0.9020 · FRR 7.14 % · FAR 2.77 %, élu : seed 42).
- **Banc streaming** (11,5 min, 23 occurrences, seuil 0.8) : **rappel 43.5 %
  (10/23) · 26.2 FA/h** — contre v03_replica : 73.9 % (17/23) · 47.1 FA/h.
  FA/h −44 % ✅, mais rappel −30 points ❌ (critère : −2 max). À 0.5, v04 reste
  en dessous (52.2 % · 36.6 FA/h) : le compromis entier s'est déplacé, ce n'est
  pas un effet de seuil.
- **Verdict** : hypothèse à moitié confirmée — la parole continue en négatif
  écrase bien les FA, mais à cette dose (1500 fenêtres) elle
  rend le modèle sourd aux vraies occurrences. Piste suivante : réduire la dose
  et/ou passer par des hard negatives ciblés (les FA réelles du banc) plutôt
  qu'un volume massif indifférencié. Preuves :
  `artifacts/runs/eloquence/v04_speech_neg/` et
  `artifacts/reports/stream_bench/eloquence.json`.

### v03_replica — 2026-08-12 — ✅ point de contrôle de la migration validé
- **But** : rejouer la recette v03 dans la nouvelle architecture. Attendu :
  F1 ≈ 0.926, FRR ≈ 5 %, FAR ≈ 2,5 %.
- **Test clips (seuil 0.5)** : F1 0.9202 · FRR 5.46 % · FAR 2.68 % · AUC 0.9915
  (élu : seed 46 par la validation, 5 candidats). Écart avec v03 largement sous
  le bruit CPU de ±0.03-0.06 (`ADR-003`).
- **Banc streaming** (11,5 min réelles, 23 occurrences, seuil 0.8) :
  **rappel 73.9 % (17/23) · 47.1 FA/h** — contre v03 rejoué sur le même banc :
  69.6 % (16/23) · 52.3 FA/h. Différence d'une occurrence et d'une FA : de la
  variance, pas un gain.
- **Verdict** : la migration reproduit la recette v03 aux fluctuations près.
  Pas de promotion (ce run est un contrôle, pas une hypothèse). Preuves :
  `artifacts/runs/eloquence/v03_replica/` et
  `artifacts/reports/dashboards/eloquence_2026-08-12.html`.

---

## Historique repris de `coach-vocal_etape1`

### v03 — 2026-07-21 — ⭐ champion à la migration
- **Nouveauté** : +500 positifs TTS Piper (dose optimale du sweep dose-réponse ;
  à 2000 le synthétique noie le réel). Protocole 5 candidats élus par la validation.
- **Test clips (seuil 0.5)** : F1 0.9262 · FRR 5.04 % · FAR 2.48 % · AUC 0.9918.
- **Banc streaming** (16 min, 38 occurrences, seuil 0.8) : **rappel 68.4 % ·
  52.7 FA/h** — meilleur des trois finalistes (v02 : 65.8 % · 56.5 ;
  candidat seed 44 : 60.5 % · 64.0).
- **Verdict** : promu sur les faits du banc. Enseignement principal : le test par
  clips classe mal (le meilleur en clip finit dernier en réel) et le déficit n°1
  est le manque de négatifs de parole continue.

### v02 — 2026-07-20
- **Nouveauté** : augmentation de vitesse ×0.85–1.15 et clips guidés dans le train.
- **Test clips** : F1 0.9516 · FRR 5.04 % · FAR 1.14 % · AUC 0.9922.
- **Verdict** : promu. FAR presque divisée par deux vs v01, robustesse au tempo
  réel passée de 6/20 à 20/20.

### v01 — 2026-07-19 — première référence
- CNN binaire, négatifs Common Voice + GSC + MUSAN + silence + fragments.
- **Test clips** : F1 0.9407 · FRR 3.36 % · FAR 2.17 % · AUC 0.9934.
- **Points faibles** : `fragments_moi` et `proches`, sanity au tempo réel 6/20.

---

## Backlog

- [x] Réplique v03 dans la nouvelle architecture (point de contrôle de migration) — validé le 2026-08-12.
- [x] v04 : négatifs de parole continue — testé le 2026-08-12, critère non
  atteint (rappel −30 pts). Reste à trouver la bonne dose ou passer aux hard
  negatives ciblés.
- [ ] Réinjecter les fausses alarmes réelles confirmées à l'oreille (hard negatives).
- [ ] Étendre le banc (60 min, vidéos non thématiques) pour des FA/h représentatives.
- [ ] Comparer `dscnn` à `cnn_baseline` au banc (et non sur la F1 par clip).
- [ ] Augmentation bruit MUSAN mélangé aux positifs à SNR variable.
- [ ] Reprendre les 8 clips guidés écartés, normalisés en gain, boost réduit.
