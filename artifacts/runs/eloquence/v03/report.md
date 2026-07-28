# Run v03 — 14_eloquence_v09_train (+ sélection v10) — 2026-07-21 23:11

Modèle élu : candidat mseed43 (13 epochs, val_loss 0.0532).

```
============================================================
TEST — éloquence v03 (seuil 0.5)
============================================================
               precision    recall  f1-score   support

pas_eloquence       0.99      0.98      0.98       969
    eloquence       0.90      0.95      0.93       238

     accuracy                           0.97      1207
    macro avg       0.95      0.96      0.95      1207
 weighted avg       0.97      0.97      0.97      1207

    FRR (éloquence ratée)      : 5.04%
    FAR (acceptation à tort)   : 2.48%
    F1 (classe éloquence)      : 0.9262
    ROC-AUC                    : 0.9918

Breakdown par pool (test) — proba moyenne / % au-dessus du seuil :
    cv_en           n=98    proba moy   3.2%   déclenche   1.0%  ✅
    cv_fr           n=195   proba moy   1.9%   déclenche   0.5%  ✅
    fragments_moi   n=30    proba moy   7.6%   déclenche   6.7%  ❌
    fragments_yt    n=126   proba moy   0.0%   déclenche   0.0%  ✅
    gsc             n=250   proba moy   1.9%   déclenche   2.0%  ✅
    moi_positif     n=5     proba moy 100.0%   déclenche 100.0%  ✅
    musan_noise     n=150   proba moy   1.4%   déclenche   0.7%  ✅
    proches         n=4     proba moy  32.5%   déclenche  25.0%  ❌
    silence         n=20    proba moy   0.0%   déclenche   0.0%  ✅
    yt_positif      n=188   proba moy  94.2%   déclenche  93.6%  ✅

Balayage du seuil (test) — pour choisir le seuil live :
     seuil      FRR      FAR
      0.30    2.94%    4.85%
      0.40    3.78%    4.33%
      0.50    5.04%    2.48%
      0.60    5.04%    1.96%
      0.70    5.04%    1.86%
      0.80    6.30%    1.44%
```

## Sélection multi-seeds (variance CPU, cf. JOURNAL 2026-07-21)

Critère : meilleure val_loss (JAMAIS le test). Candidats :

| seed | val_loss | val_acc | F1 test | FRR | FAR |
|---|---|---|---|---|---|
| 42 | 0.0761 | 97.86% | 0.9269 | 6.72% | 1.96% |
| 43 ⭐ | 0.0532 | 98.36% | 0.9262 | 5.04% | 2.48% |
| 44 | 0.0560 | 99.18% | 0.9615 | 5.46% | 0.52% |
| 45 | 0.0943 | 97.62% | 0.9419 | 4.62% | 1.75% |
| 46 | 0.0566 | 98.52% | 0.9489 | 6.30% | 0.93% |

Artefacts : model.keras, metrics.json, config.json, manifest.csv,
learning_curve.png, confusion.png, candidates/ (audit des 5 seeds)
