# Run `v04_speech_neg` — expérience `v04_speech_neg`

tts500 + 1500 fenêtres de parole continue YouTube en négatifs (train), 150 en val et test. Même architecture et même protocole que v03.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg** (seed données 42, empreinte `a1867ef00cd975c6`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 18/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 9220 | 1:3.6 |
| val | 238 | 1129 | 1:4.7 |
| test | 238 | 1119 | 1:4.7 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.46% |
| F1 (classe positive) | 0.9020 |
| **FRR** (mot raté) | 7.14% |
| **FAR** (fausse alarme) | 2.77% |
| ROC-AUC | 0.9862 |
| Clips évalués | 1357 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 ⭐ | 0.0617 | 0.9020 | 7.14% | 2.77% | 18 |
| 43 | 0.0867 | 0.8884 | 9.66% | 2.77% | 12 |
| 44 | 0.0824 | 0.8967 | 8.82% | 2.59% | 20 |
| 45 | 0.1096 | 0.8711 | 6.30% | 4.56% | 11 |
| 46 | 0.1000 | 0.8839 | 8.82% | 3.22% | 9 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
