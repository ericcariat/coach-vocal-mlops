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
