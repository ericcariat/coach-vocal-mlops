# Run `v05_metal_check` — expérience `v05_metal_check`

Recette v03_replica à l'identique, entraînée sur GPU Metal (tensorflow-metal 1.2.0) : re-contrôle d'ADR-002 et mesure du temps d'entraînement.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `3b0f0eb5888f8d4c`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 6/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **95 s** au total, dont 89.0 s de `model.fit` cumulés — backend **GPU Metal** (GPU visibles : ['/physical_device:GPU:0'])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 7720 | 1:3.0 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 89.98% |
| F1 (classe positive) | 0.7903 |
| **FRR** (mot raté) | 4.20% |
| **FAR** (fausse alarme) | 11.46% |
| ROC-AUC | 0.9480 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 2.4950 | 0.6054 | 19.75% | 20.85% | 6 |
| 43 | 0.4582 | 0.8364 | 3.36% | 8.46% | 6 |
| 44 ⭐ | 0.3642 | 0.7903 | 4.20% | 11.46% | 6 |
| 45 | 1.1044 | 0.7607 | 13.87% | 9.91% | 6 |
| 46 | 1.4819 | 0.6667 | 19.75% | 14.86% | 6 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
