# Run `v09_gate` — expérience `v09_gate`

Recette v03 (tts500) filtrée par la porte qualité ADR-007 (134 rejetés + 622 douteux exclus en attendant l'audit humain).


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `9b4922f4d399bcc6`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 25/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **1105 s** au total, dont 1099.3 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2150 | 7246 | 1:3.4 |
| val | 187 | 903 | 1:4.8 |
| test | 174 | 899 | 1:5.2 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 97.95% |
| F1 (classe positive) | 0.9385 |
| **FRR** (mot raté) | 3.45% |
| **FAR** (fausse alarme) | 1.78% |
| ROC-AUC | 0.9955 |
| Clips évalués | 1073 (dont 174 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0638 | 0.9062 | 2.87% | 3.34% | 17 |
| 43 | 0.0405 | 0.9304 | 4.02% | 2.00% | 16 |
| 44 | 0.0723 | 0.8997 | 9.77% | 2.00% | 12 |
| 45 ⭐ | 0.0155 | 0.9385 | 3.45% | 1.78% | 25 |
| 46 | 0.0599 | 0.9070 | 7.47% | 2.22% | 11 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
