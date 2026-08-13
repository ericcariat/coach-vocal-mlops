# Run `v15b_summre_150` — expérience `v15b_summre_150`

Champion + 150 négatifs de réunions SUMM-RE (train) — point 150 du sweep de dose du domaine réunion.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_300_summre_150** (seed données 42, empreinte `9c5224fdc134ef2b`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 20/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1367 s** au total, dont 1330.2 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8170 | 1:3.2 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.77% |
| F1 (classe positive) | 0.9193 |
| **FRR** (mot raté) | 6.72% |
| **FAR** (fausse alarme) | 2.37% |
| ROC-AUC | 0.9899 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0651 | 0.9414 | 5.46% | 1.55% | 17 |
| 43 | 0.0770 | 0.9350 | 6.30% | 1.65% | 15 |
| 44 ⭐ | 0.0732 | 0.9193 | 6.72% | 2.37% | 20 |
| 45 | 0.0744 | 0.9454 | 5.46% | 1.34% | 22 |
| 46 | 0.0725 | 0.8992 | 6.30% | 3.61% | 18 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
