# Run `v23_sans_fragments` — expérience `v23_sans_fragments`

Diagnostic : recette v17 sans aucune source de fragments — mesure leur contribution réelle (rappel et FA/h) au banc.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v23_sans_fragments** (seed données 42, empreinte `cc9889cbd020ecc7`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 27/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1533 s** au total, dont 1492.0 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2686 | 6763 | 1:2.5 |
| val | 238 | 763 | 1:3.2 |
| test | 238 | 753 | 1:3.2 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.17% |
| F1 (classe positive) | 0.9228 |
| **FRR** (mot raté) | 4.62% |
| **FAR** (fausse alarme) | 3.59% |
| ROC-AUC | 0.9934 |
| Clips évalués | 991 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0422 | 0.9228 | 4.62% | 3.59% | 27 |
| 43 | 0.0716 | 0.9116 | 4.62% | 4.38% | 16 |
| 44 | 0.0626 | 0.9228 | 4.62% | 3.59% | 19 |
| 45 | 0.0517 | 0.9234 | 3.78% | 3.85% | 26 |
| 46 | 0.0530 | 0.9383 | 4.20% | 2.66% | 25 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
