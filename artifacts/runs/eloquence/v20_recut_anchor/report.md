# Run `v20_recut_anchor` — expérience `v20_recut_anchor`

Positifs re-découpés sans troncature : recut yt (contexte réel + jitter 0-200 ms) + ré-ancrage moi_, time_shift 0. Seule variable vs v19_fragments.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v20_recut_anchor** (seed données 42, empreinte `74f1946845d865f6`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 10/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **764 s** au total, dont 723.2 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2667 | 8433 | 1:3.2 |
| val | 235 | 979 | 1:4.2 |
| test | 235 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.51% |
| F1 (classe positive) | 0.9095 |
| **FRR** (mot raté) | 10.21% |
| **FAR** (fausse alarme) | 1.86% |
| ROC-AUC | 0.9868 |
| Clips évalués | 1204 (dont 235 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.1003 | 0.9198 | 7.23% | 2.17% | 9 |
| 43 | 0.0932 | 0.8971 | 7.23% | 3.41% | 10 |
| 44 | 0.1344 | 0.9024 | 5.53% | 3.61% | 9 |
| 45 ⭐ | 0.0972 | 0.9095 | 10.21% | 1.86% | 10 |
| 46 | 0.1210 | 0.9024 | 5.53% | 3.61% | 9 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
