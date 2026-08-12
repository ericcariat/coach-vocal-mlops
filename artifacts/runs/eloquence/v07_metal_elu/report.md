# Run `v07_metal_elu` — expérience `v07_metal_elu`

Recette v03 (tts500), CNN de référence en elu, entraînement sur GPU Metal : deuxième essai du contournement du bug ReLU après l'échec de leaky_relu.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `3b0f0eb5888f8d4c`)
- modèle : **cnn_elu** (`cnn_baseline`)
- entraînement : 18/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **510 s** au total, dont 503.6 s de `model.fit` cumulés — backend **GPU Metal** (GPU visibles : ['/physical_device:GPU:0'])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 7720 | 1:3.0 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 97.18% |
| F1 (classe positive) | 0.9286 |
| **FRR** (mot raté) | 7.14% |
| **FAR** (fausse alarme) | 1.75% |
| ROC-AUC | 0.9859 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0815 | 0.9286 | 7.14% | 1.75% | 18 |
| 43 | 0.1012 | 0.9339 | 5.04% | 2.06% | 14 |
| 44 | 0.0939 | 0.8802 | 5.88% | 4.85% | 9 |
| 45 | 0.0837 | 0.8967 | 8.82% | 2.99% | 11 |
| 46 | 0.0903 | 0.8827 | 6.72% | 4.44% | 10 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
