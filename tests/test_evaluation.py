"""Métriques : FRR/FAR ne doivent jamais être confondues, et l'appariement
du banc streaming doit être conservateur (dans le doute, on ne compte pas)."""

from __future__ import annotations

import numpy as np

from coachvocal.evaluation.clip_eval import evaluate, metrics_at
from coachvocal.evaluation.stream_bench import score


def test_frr_et_far_ne_sont_pas_symetriques():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.9, 0.2, 0.9, 0.1, 0.1, 0.7, 0.1])
    m = metrics_at(y_true, y_proba, 0.5)
    assert m["frr"] == 0.25         # 1 positif raté sur 4
    assert m["far"] == 0.25         # 1 négatif accepté sur 4
    assert m["fn"] == 1 and m["fp"] == 1


def test_accuracy_trompeuse_sur_donnees_desequilibrees():
    """40 négatifs pour 1 positif : tout prédire à « non » donne 97,6 %
    d'accuracy et une F1 nulle. C'est pourquoi on ne décide jamais sur l'accuracy."""
    y_true = np.array([1] + [0] * 40)
    y_proba = np.zeros(41)
    m = evaluate(y_true, y_proba, 0.5)
    assert m["accuracy"] > 0.97
    assert m["f1_pos"] == 0.0
    assert m["frr"] == 1.0


def test_balayage_de_seuil_monotone():
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 50 + [0] * 50)
    y_proba = np.concatenate([rng.uniform(0.5, 1, 50), rng.uniform(0, 0.5, 50)])
    sweep = evaluate(y_true, y_proba, 0.5)["threshold_sweep"]
    frr = [s["frr"] for s in sweep]
    far = [s["far"] for s in sweep]
    assert frr == sorted(frr)              # seuil ↑ → on rate davantage
    assert far == sorted(far, reverse=True)  # seuil ↑ → moins de fausses alarmes


def test_appariement_streaming():
    # occurrence à 10 s ; fenêtre d'appariement [-0.5 s, +2.0 s]
    r = score(triggers=[10.2], occurrences=[10.0], uncertain=[])
    assert r["detected"] == 1 and r["false_alarms"] == 0


def test_trigger_isole_compte_comme_fausse_alarme():
    r = score(triggers=[30.0], occurrences=[10.0], uncertain=[])
    assert r["false_alarms"] == 1 and r["detected"] == 0


def test_zone_incertaine_ni_detection_ni_fausse_alarme():
    """Vérité terrain douteuse (VTT qui dérive) : on refuse de trancher plutôt
    que de gonfler artificiellement les FA/h."""
    r = score(triggers=[30.0], occurrences=[10.0], uncertain=[31.0])
    assert r["false_alarms"] == 0 and r["uncertain"] == 1


def test_occurrence_ratee_produit_un_fn():
    r = score(triggers=[], occurrences=[10.0], uncertain=[])
    assert r["detected"] == 0
    assert any(e["kind"] == "FN" for e in r["events"])
