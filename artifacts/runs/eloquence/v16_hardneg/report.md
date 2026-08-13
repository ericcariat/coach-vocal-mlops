# Run `v16_hardneg` — expérience `v16_hardneg`

Champion + hard negatives (18 FA YouTube confirmées à l'oreille, ×3 fenêtres, boost 10) + élection fa_ambient — la boucle banc → humain → entraînement.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_300_hn** (seed données 42, empreinte `63b7abff2b04a8e4`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 15/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1012 s** au total, dont 973.9 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8560 | 1:3.3 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.85% |
| F1 (classe positive) | 0.9181 |
| **FRR** (mot raté) | 10.50% |
| **FAR** (fausse alarme) | 1.34% |
| ROC-AUC | 0.9855 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0663 | 0.9181 | 10.50% | 1.34% | 15 |
| 43 | 0.0754 | 0.8595 | 11.34% | 4.33% | 12 |
| 44 | 0.1048 | 0.8758 | 9.66% | 3.92% | 13 |
| 45 | 0.1070 | 0.8956 | 6.30% | 3.82% | 12 |
| 46 | 0.0874 | 0.8689 | 10.92% | 3.92% | 13 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
