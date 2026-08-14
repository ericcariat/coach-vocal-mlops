# Run `v19_fragments` — expérience `v19_fragments`

Recette v17 avec fragments corrigés (fractions du MOT, plafond 45 % — plus de quasi-mots dans les négatifs).


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v19_fragments** (seed données 42, empreinte `3ab9d74cb0ccfde5`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 30/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1654 s** au total, dont 1609.7 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2686 | 8433 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.85% |
| F1 (classe positive) | 0.9221 |
| **FRR** (mot raté) | 5.46% |
| **FAR** (fausse alarme) | 2.58% |
| ROC-AUC | 0.9906 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0546 | 0.9221 | 5.46% | 2.58% | 30 |
| 43 | 0.0429 | 0.9224 | 5.04% | 2.68% | 20 |
| 44 | 0.0742 | 0.9405 | 3.78% | 2.06% | 14 |
| 45 | 0.0385 | 0.9356 | 5.46% | 1.86% | 26 |
| 46 | 0.0937 | 0.9209 | 4.62% | 2.89% | 12 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
