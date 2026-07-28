# Run v01 — 14_eloquence_v02_train — 2026-07-19 21:15

Epochs : 10/30 (EarlyStopping) — acc train 99.19% / val 97.95%

```
============================================================
TEST — éloquence v01 (seuil 0.5)
============================================================
               precision    recall  f1-score   support

pas_eloquence       0.99      0.98      0.98       969
    eloquence       0.92      0.97      0.94       238

     accuracy                           0.98      1207
    macro avg       0.95      0.97      0.96      1207
 weighted avg       0.98      0.98      0.98      1207

    FRR (éloquence ratée)      : 3.36%
    FAR (acceptation à tort)   : 2.17%
    F1 (classe éloquence)      : 0.9407
    ROC-AUC                    : 0.9934

Breakdown par pool (test) — proba moyenne / % au-dessus du seuil :
    cv_en           n=98    proba moy   2.3%   déclenche   2.0%  ✅
    cv_fr           n=195   proba moy   0.9%   déclenche   0.5%  ✅
    fragments_moi   n=30    proba moy  10.9%   déclenche   6.7%  ❌
    fragments_yt    n=126   proba moy   0.1%   déclenche   0.0%  ✅
    gsc             n=250   proba moy   0.6%   déclenche   0.4%  ✅
    moi_positif     n=5     proba moy 100.0%   déclenche 100.0%  ✅
    musan_noise     n=150   proba moy   1.8%   déclenche   0.7%  ✅
    proches         n=4     proba moy  17.0%   déclenche  25.0%  ❌
    silence         n=20    proba moy   0.0%   déclenche   0.0%  ✅
    yt_positif      n=188   proba moy  94.6%   déclenche  95.7%  ✅
```

Artefacts : model.keras, metrics.json, config.json, manifest.csv,
learning_curve.png, confusion.png
