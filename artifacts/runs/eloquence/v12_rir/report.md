# Run `v12_rir` — expérience `v12_rir`

Recette v03 + augmentation réverbération RIR (p=0.5) et bruit multi-SNR 5-20 dB (p=0.5) — combler l'écart d'augmentation avec l'état de l'art.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `3b0f0eb5888f8d4c`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 28/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **1395 s** au total, dont 1385.1 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

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
| F1 (classe positive) | 0.9303 |
| **FRR** (mot raté) | 4.62% |
| **FAR** (fausse alarme) | 2.37% |
| ROC-AUC | 0.9912 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0647 | 0.9234 | 3.78% | 2.99% | 27 |
| 43 ⭐ | 0.0498 | 0.9303 | 4.62% | 2.37% | 28 |
| 44 | 0.0895 | 0.9121 | 6.30% | 2.89% | 16 |
| 45 | 0.0771 | 0.9243 | 5.04% | 2.58% | 15 |
| 46 | 0.0740 | 0.8867 | 6.30% | 4.33% | 12 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
