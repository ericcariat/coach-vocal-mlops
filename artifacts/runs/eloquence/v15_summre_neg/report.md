# Run `v15_summre_neg` — expérience `v15_summre_neg`

Champion + 300 négatifs de réunions SUMM-RE (train, réunions disjointes du banc et de val_ambient) + élection fa_ambient — cibler les FA du domaine réunion.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_300_summre** (seed données 42, empreinte `7f83e53a2afee7bf`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 14/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1176 s** au total, dont 1136.5 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8320 | 1:3.2 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 97.27% |
| F1 (classe positive) | 0.9299 |
| **FRR** (mot raté) | 7.98% |
| **FAR** (fausse alarme) | 1.44% |
| ROC-AUC | 0.9905 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0612 | 0.9299 | 7.98% | 1.44% | 14 |
| 43 | 0.0495 | 0.9378 | 5.04% | 1.86% | 20 |
| 44 | 0.0939 | 0.9153 | 4.62% | 3.20% | 12 |
| 45 | 0.0591 | 0.9571 | 6.30% | 0.52% | 16 |
| 46 | 0.0702 | 0.9117 | 6.72% | 2.79% | 15 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
