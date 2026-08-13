# Run `v17_stack` — expérience `v17_stack`

Cumul rappel-d'abord : RIR/multi-SNR + dose 300 YouTube + SUMM-RE 75 + hard negatives ×2, élu par fa_ambient.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v17_stack** (seed données 42, empreinte `8e59e5077b0c34bb`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 25/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1199 s** au total, dont 1160.5 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8203 | 1:3.2 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.69% |
| F1 (classe positive) | 0.9180 |
| **FRR** (mot raté) | 5.88% |
| **FAR** (fausse alarme) | 2.68% |
| ROC-AUC | 0.9890 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0597 | 0.9180 | 5.88% | 2.68% | 25 |
| 43 | 0.0963 | 0.8715 | 8.82% | 4.44% | 13 |
| 44 | 0.0962 | 0.8970 | 6.72% | 3.61% | 12 |
| 45 | 0.0852 | 0.8898 | 5.04% | 4.54% | 12 |
| 46 | 0.0696 | 0.8996 | 5.88% | 3.72% | 17 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
