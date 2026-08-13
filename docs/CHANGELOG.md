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

### v10_recut — 2026-08-13 — ❌ instructif : la géométrie seule casse tout
- **But** : positifs re-découpés fin-de-fenêtre + jitter (levier n°1 de la
  littérature), une seule variable vs v03.
- **Banc étendu (seuil 0.8)** : rappel **100 %** (25/25, toutes formes) mais
  **781.9 FA/h** — le modèle tire sur tout.
- **Diagnostic** : les positifs re-découpés (contexte réel + mot) sont les
  SEULS exemples d'entraînement ressemblant à un flux naturel ; face à des
  négatifs restés « mots isolés / crops d'1 s », le modèle apprend « parole
  continue = positif ». **La re-découpe et les négatifs de parole continue
  sont un couple** — v10b les teste ensemble. Rétrospectivement, ceci éclaire
  aussi v04 : sa chute de rappel était l'image miroir du même déséquilibre.

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
