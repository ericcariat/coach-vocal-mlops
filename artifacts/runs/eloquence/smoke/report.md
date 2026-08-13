# Run `smoke` — expérience `smoke`

Contrôle de bout en bout du pipeline (résultats sans valeur scientifique).

## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **smoke** (seed données 42, empreinte `8ecbf4603ae3e769`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 2/2 epochs, batch 64, lr 0.001, seeds [42] (élu par `val_loss`)
- durée : **23 s** au total, dont 20.5 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2076 | 5140 | 1:2.5 |
| val | 238 | 671 | 1:2.8 |
| test | 238 | 661 | 1:2.8 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 91.88% |
| F1 (classe positive) | 0.8610 |
| **FRR** (mot raté) | 5.04% |
| **FAR** (fausse alarme) | 9.23% |
| ROC-AUC | 0.9710 |
| Clips évalués | 899 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.2548 | 0.8610 | 5.04% | 9.23% | 2 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
