# Run `smoke` — expérience `smoke`

Contrôle de bout en bout du pipeline (résultats sans valeur scientifique).

## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **smoke** (seed données 42, empreinte `3950cdfe228ddd07`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 2/2 epochs, batch 64, lr 0.001, seeds [42] (élu par `val_loss`)
- durée : **24 s** au total, dont 21.1 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2186 | 5370 | 1:2.5 |
| val | 238 | 671 | 1:2.8 |
| test | 238 | 661 | 1:2.8 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 90.88% |
| F1 (classe positive) | 0.8487 |
| **FRR** (mot raté) | 3.36% |
| **FAR** (fausse alarme) | 11.20% |
| ROC-AUC | 0.9771 |
| Clips évalués | 899 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.2719 | 0.8487 | 3.36% | 11.20% | 2 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
