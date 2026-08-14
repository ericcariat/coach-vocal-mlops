# Run `v24_fragments_longs` — expérience `v24_fragments_longs`

Recette v17 avec fragments propres à plafond 70 % du mot (longs mais jamais complets) — récupérer le garde-fou anti-FA sans quasi-mots étiquetés négatifs.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **v24_fragments_longs** (seed données 42, empreinte `b31d85ad88224a92`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 27/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `fa_ambient`)
- durée : **1629 s** au total, dont 1590.3 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2686 | 8433 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.93% |
| F1 (classe positive) | 0.9240 |
| **FRR** (mot raté) | 5.46% |
| **FAR** (fausse alarme) | 2.48% |
| ROC-AUC | 0.9915 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0689 | 0.9146 | 5.46% | 2.99% | 21 |
| 43 | 0.0560 | 0.9434 | 5.46% | 1.44% | 16 |
| 44 | 0.0717 | 0.9339 | 5.04% | 2.06% | 22 |
| 45 ⭐ | 0.0293 | 0.9240 | 5.46% | 2.48% | 27 |
| 46 | 0.0806 | 0.9361 | 4.62% | 2.06% | 18 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
