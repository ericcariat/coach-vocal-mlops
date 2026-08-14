# Run `v21_fragments_propres` — expérience `v21_fragments_propres`

Recette v17 + fragments réellement propres (mesure du mot par énergie relative au pic, découpe depuis les bornes du mot, plafond 45 %).


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v19_fragments** (seed données 42, empreinte `3ab9d74cb0ccfde5`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 21/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1540 s** au total, dont 1499.8 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2686 | 8433 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.77% |
| F1 (classe positive) | 0.9206 |
| **FRR** (mot raté) | 5.04% |
| **FAR** (fausse alarme) | 2.79% |
| ROC-AUC | 0.9893 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0532 | 0.9187 | 5.04% | 2.89% | 22 |
| 43 ⭐ | 0.0457 | 0.9206 | 5.04% | 2.79% | 21 |
| 44 | 0.0675 | 0.9194 | 4.20% | 3.10% | 13 |
| 45 | 0.0422 | 0.9314 | 5.88% | 1.96% | 23 |
| 46 | 0.0799 | 0.9202 | 5.46% | 2.68% | 19 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
