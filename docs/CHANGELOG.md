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

### v04_speech_neg — *à lancer*
- **Hypothèse** : les négatifs de parole continue font baisser les FA/heure.
- **Critère défini à l'avance** : FA/h −20 % minimum, rappel −2 points maximum.
- **Configuration** : `configs/experiment/v04_speech_neg.yaml`.

### v03_replica — *à lancer* (réplique de v03 dans la nouvelle architecture)
- **But** : point de contrôle de la migration. On attend les mêmes ordres de
  grandeur que le v03 historique (F1 ≈ 0.926, FRR ≈ 5 %, FAR ≈ 2,5 %).
- Un écart important signalerait une différence de pipeline, pas de recette —
  à investiguer avant tout nouveau run.

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

- [ ] Réplique v03 dans la nouvelle architecture (point de contrôle de migration).
- [ ] v04 : négatifs de parole continue — déficit n°1 révélé par le banc.
- [ ] Réinjecter les fausses alarmes réelles confirmées à l'oreille (hard negatives).
- [ ] Étendre le banc (60 min, vidéos non thématiques) pour des FA/h représentatives.
- [ ] Comparer `dscnn` à `cnn_baseline` au banc (et non sur la F1 par clip).
- [ ] Augmentation bruit MUSAN mélangé aux positifs à SNR variable.
- [ ] Reprendre les 8 clips guidés écartés, normalisés en gain, boost réduit.
