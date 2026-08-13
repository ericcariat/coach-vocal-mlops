# Run `v11_speech_300` — expérience `v11_speech_300`

tts500 + 300 fenêtres de parole continue en négatif (train seulement) — point 300 du sweep dose-réponse.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_300** (seed données 42, empreinte `afb0981e4e020c30`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 22/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **1089 s** au total, dont 1081.3 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8020 | 1:3.1 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 97.43% |
| F1 (classe positive) | 0.9347 |
| **FRR** (mot raté) | 6.72% |
| **FAR** (fausse alarme) | 1.55% |
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
| 42 | 0.0724 | 0.9020 | 7.14% | 3.20% | 10 |
| 43 ⭐ | 0.0413 | 0.9347 | 6.72% | 1.55% | 22 |
| 44 | 0.0735 | 0.9353 | 5.88% | 1.75% | 14 |
| 45 | 0.0723 | 0.9395 | 5.46% | 1.65% | 13 |
| 46 | 0.0590 | 0.9320 | 5.04% | 2.17% | 17 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
