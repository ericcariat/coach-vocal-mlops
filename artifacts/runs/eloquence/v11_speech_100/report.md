# Run `v11_speech_100` — expérience `v11_speech_100`

tts500 + 100 fenêtres de parole continue en négatif (train seulement) — point 100 du sweep dose-réponse.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_100** (seed données 42, empreinte `0b861ed5a25c95c6`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 15/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **1008 s** au total, dont 1000.4 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 7820 | 1:3.0 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 97.76% |
| F1 (classe positive) | 0.9439 |
| **FRR** (mot raté) | 4.62% |
| **FAR** (fausse alarme) | 1.65% |
| ROC-AUC | 0.9891 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0971 | 0.9106 | 5.88% | 3.10% | 14 |
| 43 | 0.0601 | 0.9300 | 5.04% | 2.27% | 14 |
| 44 | 0.0692 | 0.9287 | 4.20% | 2.58% | 16 |
| 45 | 0.0863 | 0.8996 | 5.88% | 3.72% | 11 |
| 46 ⭐ | 0.0590 | 0.9439 | 4.62% | 1.65% | 15 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
