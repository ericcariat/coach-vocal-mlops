# Run `v10b_recut_speech` — expérience `v10b_recut_speech`

Positifs re-découpés fin-de-fenêtre + 300 négatifs de parole continue — le couple géométrie/contrepoids, après l'échec instructif de v10 seul.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500_recut_sn300** (seed données 42, empreinte `57b3f53a692e211f`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 14/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **966 s** au total, dont 954.6 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2347 | 8020 | 1:3.4 |
| val | 213 | 979 | 1:4.6 |
| test | 211 | 969 | 1:4.6 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 95.17% |
| F1 (classe positive) | 0.8747 |
| **FRR** (mot raté) | 5.69% |
| **FAR** (fausse alarme) | 4.64% |
| ROC-AUC | 0.9872 |
| Clips évalués | 1180 (dont 211 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.1515 | 0.8596 | 4.27% | 5.88% | 9 |
| 43 | 0.1447 | 0.8894 | 4.74% | 4.13% | 21 |
| 44 | 0.1407 | 0.8874 | 6.64% | 3.72% | 14 |
| 45 ⭐ | 0.1245 | 0.8747 | 5.69% | 4.64% | 14 |
| 46 | 0.1417 | 0.8758 | 8.06% | 3.92% | 8 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
