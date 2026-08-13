# Run `v11_speech_500` — expérience `v11_speech_500`

tts500 + 500 fenêtres de parole continue en négatif (train seulement) — point 500 du sweep dose-réponse.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **speech_neg_500** (seed données 42, empreinte `d183c7a819801e84`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 11/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **1212 s** au total, dont 1201.2 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2576 | 8220 | 1:3.2 |
| val | 238 | 979 | 1:4.1 |
| test | 238 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.11% |
| F1 (classe positive) | 0.9066 |
| **FRR** (mot raté) | 4.20% |
| **FAR** (fausse alarme) | 3.82% |
| ROC-AUC | 0.9887 |
| Clips évalués | 1207 (dont 238 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0735 | 0.9499 | 8.40% | 0.31% | 21 |
| 43 | 0.0567 | 0.9087 | 7.98% | 2.58% | 14 |
| 44 | 0.0847 | 0.9136 | 6.72% | 2.68% | 16 |
| 45 | 0.0766 | 0.9389 | 6.30% | 1.44% | 21 |
| 46 ⭐ | 0.0519 | 0.9066 | 4.20% | 3.82% | 11 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
