# Run v02 — 14_eloquence_v05_train — 2026-07-20 00:42

Epochs : 20/30 (EarlyStopping) — acc train 98.58% / val 98.36%

```
============================================================
TEST — éloquence v02 (seuil 0.5)
============================================================
               precision    recall  f1-score   support

pas_eloquence       0.99      0.99      0.99       969
    eloquence       0.95      0.95      0.95       238

     accuracy                           0.98      1207
    macro avg       0.97      0.97      0.97      1207
 weighted avg       0.98      0.98      0.98      1207

    FRR (éloquence ratée)      : 5.04%
    FAR (acceptation à tort)   : 1.14%
    F1 (classe éloquence)      : 0.9516
    ROC-AUC                    : 0.9922

Breakdown par pool (test) — proba moyenne / % au-dessus du seuil :
    cv_en           n=98    proba moy   1.2%   déclenche   1.0%  ✅
    cv_fr           n=195   proba moy   0.3%   déclenche   0.0%  ✅
    fragments_moi   n=30    proba moy   6.8%   déclenche   6.7%  ❌
    fragments_yt    n=126   proba moy   0.0%   déclenche   0.0%  ✅
    gsc             n=250   proba moy   1.3%   déclenche   1.2%  ✅
    moi_positif     n=5     proba moy 100.0%   déclenche 100.0%  ✅
    musan_noise     n=150   proba moy   1.0%   déclenche   0.7%  ✅
    proches         n=4     proba moy   9.0%   déclenche   0.0%  ✅
    silence         n=20    proba moy   0.0%   déclenche   0.0%  ✅
    yt_positif      n=188   proba moy  93.7%   déclenche  93.6%  ✅

Sanity clips guidés à vitesse réelle (DANS le train, optimiste) :
    guided_003042_503_TP.wav         meilleure fenêtre 100.0% ✅
    guided_003052_049_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003102_244_TP.wav         meilleure fenêtre 100.0% ✅
    guided_003110_634_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003121_771_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003131_916_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003139_885_TP.wav         meilleure fenêtre 100.0% ✅
    guided_003147_039_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003153_762_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003204_524_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003211_879_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003218_829_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003225_595_TP.wav         meilleure fenêtre 100.0% ✅
    guided_003232_990_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003240_850_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003247_861_FN.wav         meilleure fenêtre 100.0% ✅
    guided_003259_210_FN.wav         meilleure fenêtre 100.0% ✅
    guided_234431_785_TP.wav         meilleure fenêtre 100.0% ✅
    guided_234446_108_FN.wav         meilleure fenêtre 100.0% ✅
    guided_234508_016_FN.wav         meilleure fenêtre 100.0% ✅
    → 20/20 ≥ 80% (v01 : 6/20)
```

Artefacts : model.keras, metrics.json, config.json, manifest.csv,
learning_curve.png, confusion.png
