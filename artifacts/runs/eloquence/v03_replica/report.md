# Run `v03_replica` — expérience `v03_replica`

CNN de référence + 500 positifs TTS (dose optimale du sweep). 5 candidats seeds 42-46, élu par la validation.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500** (seed données 42, empreinte `3b0f0eb5888f8d4c`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 21/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 7720 | 1:3.0 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.77% |
| F1 (classe positive) | 0.9202 |
| **FRR** (mot raté) | 5.46% |
| **FAR** (fausse alarme) | 2.68% |
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
| 42 | 0.0778 | 0.9172 | 4.62% | 3.10% | 13 |
| 43 | 0.0729 | 0.9196 | 6.30% | 2.48% | 14 |
| 44 | 0.0852 | 0.9190 | 4.62% | 2.99% | 14 |
| 45 | 0.0626 | 0.9534 | 5.46% | 0.93% | 12 |
| 46 ⭐ | 0.0599 | 0.9202 | 5.46% | 2.68% | 21 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
