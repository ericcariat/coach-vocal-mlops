# Run `v22_replica17` — expérience `v22_replica17`

Réplique de contrôle de v17_stack (aucun changement) — vérifie que le pipeline d'aujourd'hui reproduit un modèle de la classe du champion.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v17_stack** (seed données 42, empreinte `f0ad2910ea7eacc0`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 25/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1461 s** au total, dont 1419.6 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2686 | 8433 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.52% |
| F1 (classe positive) | 0.9136 |
| **FRR** (mot raté) | 6.72% |
| **FAR** (fausse alarme) | 2.68% |
| ROC-AUC | 0.9896 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0789 | 0.9370 | 6.30% | 1.55% | 23 |
| 43 | 0.0680 | 0.8871 | 5.88% | 4.44% | 18 |
| 44 | 0.1039 | 0.8986 | 5.04% | 4.02% | 15 |
| 45 ⭐ | 0.0460 | 0.9136 | 6.72% | 2.68% | 25 |
| 46 | 0.1035 | 0.9095 | 5.04% | 3.41% | 12 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
