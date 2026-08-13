# Run `v13_fa_select` — expérience `v13_fa_select`

Recette v03 élue par FA/h sur flux ambiant de validation (SUMM-RE hors banc) sous contrainte de rappel val ≥ 90 % — sélection produit, pas val_loss.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `3b0f0eb5888f8d4c`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 22/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1062 s** au total, dont 1027.4 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 7720 | 1:3.0 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 97.68% |
| F1 (classe positive) | 0.9412 |
| **FRR** (mot raté) | 5.88% |
| **FAR** (fausse alarme) | 1.44% |
| ROC-AUC | 0.9904 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0471 | 0.9412 | 5.88% | 1.44% | 22 |
| 43 | 0.0806 | 0.9259 | 5.46% | 2.37% | 17 |
| 44 | 0.1056 | 0.8959 | 4.20% | 4.44% | 10 |
| 45 | 0.0605 | 0.9548 | 6.72% | 0.52% | 13 |
| 46 | 0.0807 | 0.9281 | 5.04% | 2.37% | 11 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
