# Run `v10_recut` — expérience `v10_recut`

Recette v03 avec positifs YouTube re-découpés fin-de-fenêtre + jitter 200 ms (dataset tts500_recut). Une seule variable : la géométrie de découpe.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500_recut** (seed données 42, empreinte `a6459e94e00a0330`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 9/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **801 s** au total, dont 794.0 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2347 | 7720 | 1:3.3 |
| val | 213 | 979 | 1:4.6 |
| test | 211 | 969 | 1:4.6 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 94.83% |
| F1 (classe positive) | 0.8665 |
| **FRR** (mot raté) | 6.16% |
| **FAR** (fausse alarme) | 4.95% |
| ROC-AUC | 0.9833 |
| Clips évalués | 1180 (dont 211 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.1853 | 0.8313 | 4.27% | 7.53% | 8 |
| 43 | 0.1527 | 0.8904 | 5.69% | 3.82% | 14 |
| 44 | 0.1608 | 0.8914 | 4.74% | 4.02% | 15 |
| 45 ⭐ | 0.1244 | 0.8665 | 6.16% | 4.95% | 9 |
| 46 | 0.1299 | 0.8998 | 4.27% | 3.72% | 12 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
