# Run `v14_rir_speech300` — expérience `v14_rir_speech300`

Dose 300 de parole continue (champion v11) + augmentation RIR/multi-SNR (v12) — le cumul des deux gains orthogonaux du P1.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_300** (seed données 42, empreinte `afb0981e4e020c30`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 30/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **1425 s** au total, dont 1416.5 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8020 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.44% |
| F1 (classe positive) | 0.9128 |
| **FRR** (mot raté) | 5.46% |
| **FAR** (fausse alarme) | 3.10% |
| ROC-AUC | 0.9902 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0738 | 0.9109 | 5.46% | 3.20% | 23 |
| 43 | 0.0852 | 0.8972 | 4.62% | 4.23% | 17 |
| 44 ⭐ | 0.0560 | 0.9128 | 5.46% | 3.10% | 30 |
| 45 | 0.1047 | 0.8884 | 6.30% | 4.23% | 12 |
| 46 | 0.0869 | 0.9143 | 5.88% | 2.89% | 16 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
