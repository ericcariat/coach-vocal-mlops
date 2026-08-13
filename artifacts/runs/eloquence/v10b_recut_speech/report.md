# Run `v10b_recut_speech` — expérience `v10b_recut_speech`

Positifs re-découpés fin-de-fenêtre + 300 négatifs de parole continue — le couple géométrie/contrepoids, après l'échec instructif de v10 seul.


## Configuration

- mot-clé : **eloquence** — 16000 Hz, 40 mels, fenêtre 1.0 s
- dataset : **tts500_recut_sn300** (seed données 42, empreinte `1d0ab51314269b0f`)
- modèle : **cnn_baseline** (`cnn_baseline`)
- entraînement : 8/30 epochs, batch 64, lr 0.001, seeds [42, 43, 44, 45, 46] (élu par `val_loss`)
- durée : **625 s** au total, dont 616.9 s de `model.fit` cumulés — backend **CPU** (GPU visibles : [])

## Composition du jeu de données

| Split | Positifs | Négatifs | Ratio |
|---|---:|---:|---|
| train | 2557 | 8020 | 1:3.1 |
| val | 235 | 979 | 1:4.2 |
| test | 235 | 969 | 1:4.1 |

## Test par clips (seuil 0.5)

| Métrique | Valeur |
|---|---:|
| Accuracy | 96.59% |
| F1 (classe positive) | 0.9144 |
| **FRR** (mot raté) | 6.81% |
| **FAR** (fausse alarme) | 2.58% |
| ROC-AUC | 0.9880 |
| Clips évalués | 1204 (dont 235 positifs) |

> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La
> décision de promotion se prend au **banc streaming**
> (`coachvocal bench`), qui reproduit les conditions réelles.

## Preuves

`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`

## Candidats (tous les seeds)

| Seed | val_loss | F1 test | FRR | FAR | Epochs |
|---|---:|---:|---:|---:|---:|
| 42 | 0.1015 | 0.9106 | 4.68% | 3.41% | 12 |
| 43 | 0.0999 | 0.9155 | 5.53% | 2.89% | 9 |
| 44 | 0.1265 | 0.8755 | 10.21% | 3.72% | 6 |
| 45 | 0.1196 | 0.8952 | 5.53% | 4.02% | 8 |
| 46 ⭐ | 0.0861 | 0.9144 | 6.81% | 2.58% | 8 |

> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées
> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais
> de sélection.
