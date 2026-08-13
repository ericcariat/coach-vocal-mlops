# Run `v18_guided` — expérience `v18_guided`

v17 + essais guidés de l'auteur (TP mot nu + 15 FP cousins en négatifs durs ×5) — la boucle test guidé → entraînement.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v18_stack_guided** (seed données 42, empreinte `0958b5878a19693a`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 14/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1199 s** au total, dont 1153.8 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2686 | 8318 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 95.69% |
| F1 (classe positive) | 0.8956 |
| **FRR** (mot raté) | 6.30% |
| **FAR** (fausse alarme) | 3.82% |
| ROC-AUC | 0.9864 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0833 | 0.8956 | 6.30% | 3.82% | 14 |
| 43 | 0.0958 | 0.8867 | 4.62% | 4.85% | 14 |
| 44 | 0.0894 | 0.9162 | 5.88% | 2.79% | 20 |
| 45 | 0.0893 | 0.8928 | 3.78% | 4.75% | 15 |
| 46 | 0.1036 | 0.8897 | 3.36% | 5.06% | 12 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
