# Run `v06_metal_leaky` — expérience `v06_metal_leaky`

Recette v03 (tts500), CNN de référence en leaky_relu, entraînement sur GPU Metal : le contournement du bug ReLU rend-il Metal utilisable, et à quelle vitesse ?


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `3b0f0eb5888f8d4c`)
- modèle : **cnn_leaky** (`cnn_baseline`)
- entraînement : 9/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **159 s** au total, dont 153.1 s de `model.fit` cumulés — backend **GPU Metal** (GPU visibles : ['/physical_device:GPU:0'])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 7720 | 1:3.0 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.35% |
| F1 (classe positive) | 0.9120 |
| **FRR** (mot raté) | 4.20% |
| **FAR** (fausse alarme) | 3.51% |
| ROC-AUC | 0.9838 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.1600 | 0.8730 | 7.56% | 4.75% | 7 |
| 43 | 0.1444 | 0.9363 | 4.20% | 2.17% | 19 |
| 44 | 0.1334 | 0.8907 | 5.88% | 4.23% | 9 |
| 45 | 0.1436 | 0.8672 | 6.72% | 5.37% | 8 |
| 46 ⭐ | 0.1219 | 0.9120 | 4.20% | 3.51% | 9 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
