# Run `v15b_summre_75` — expérience `v15b_summre_75`

Champion + 75 négatifs de réunions SUMM-RE (train) — point 75 du sweep de dose du domaine réunion.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_300_summre_75** (seed données 42, empreinte `0148de63a0c3df28`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 14/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **924 s** au total, dont 883.8 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8095 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.11% |
| F1 (classe positive) | 0.9023 |
| **FRR** (mot raté) | 8.82% |
| **FAR** (fausse alarme) | 2.68% |
| ROC-AUC | 0.9880 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0705 | 0.9250 | 6.72% | 2.06% | 14 |
| 43 | 0.1191 | 0.8843 | 10.08% | 3.30% | 9 |
| 44 ⭐ | 0.0544 | 0.9023 | 8.82% | 2.68% | 14 |
| 45 | 0.1220 | 0.9069 | 5.88% | 3.30% | 12 |
| 46 | 0.0745 | 0.9061 | 6.72% | 3.10% | 11 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
