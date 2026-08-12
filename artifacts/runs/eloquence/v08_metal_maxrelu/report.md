# Run `v08_metal_maxrelu` — expérience `v08_metal_maxrelu`

Recette v03 (tts500), CNN identique à la référence via relu_max (≡ ReLU, non fusionné), entraînement sur GPU Metal : le test qui sépare l'effet backend de l'effet activation.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `3b0f0eb5888f8d4c`)
- modèle : **cnn_maxrelu** (`cnn_baseline`)
- entraînement : 16/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **355 s** au total, dont 349.7 s de `model.fit` cumulés — backend **GPU Metal** (GPU visibles : ['/physical_device:GPU:0'])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 7720 | 1:3.0 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 97.27% |
| F1 (classe positive) | 0.9322 |
| **FRR** (mot raté) | 4.62% |
| **FAR** (fausse alarme) | 2.27% |
| ROC-AUC | 0.9898 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0563 | 0.9322 | 4.62% | 2.27% | 16 |
| 43 | 0.0976 | 0.9128 | 5.46% | 3.10% | 11 |
| 44 | 0.1024 | 0.9153 | 4.62% | 3.20% | 12 |
| 45 | 0.0578 | 0.9552 | 5.88% | 0.72% | 17 |
| 46 | 0.0625 | 0.9500 | 4.20% | 1.44% | 18 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
