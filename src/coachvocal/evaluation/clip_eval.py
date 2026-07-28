"""Évaluation par clip : accuracy, F1, FRR, FAR, AUC, balayage de seuil.

Vocabulaire du domaine, à ne pas confondre :
- **FRR** (False Rejection Rate) = 1 − rappel de la classe positive : le mot est
  prononcé, le détecteur ne réagit pas. Ce qui agace l'utilisateur.
- **FAR** (False Acceptance Rate) = taux de négatifs acceptés : le détecteur se
  réveille tout seul. Ce qui rend un always-on inutilisable.

⚠️ Ces chiffres décrivent une population de clips de 1 s pré-découpés, pas la
production. Le banc streaming (`stream_bench.py`) a montré qu'ils **classent mal**
les modèles : le meilleur en clip était le pire en réel. On les garde comme
signal de contrôle, la décision de promotion revient au banc.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def metrics_at(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba > threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()),
        "precision_pos": float(pr[1]),
        "recall_pos": float(rc[1]),
        "f1_pos": float(f1[1]),
        "frr": float(1 - rc[1]),          # positifs ratés
        "far": float(1 - rc[0]),          # négatifs acceptés
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).astype(float).ravel()
    out = metrics_at(y_true, y_proba, threshold)
    out["n"] = int(len(y_true))
    out["n_pos"] = int(y_true.sum())
    out["roc_auc"] = float(roc_auc_score(y_true, y_proba)) if 0 < y_true.sum() < len(y_true) else float("nan")
    out["threshold_sweep"] = [metrics_at(y_true, y_proba, t) for t in THRESHOLD_GRID]
    return out


def pool_breakdown(manifest, probas_by_file: dict[str, float], split: str = "test",
                   threshold: float = 0.5) -> dict[str, dict]:
    """Taux de déclenchement par pool : c'est ici qu'on voit QUELLE famille de
    négatifs pose problème (le pool `fragments_moi` a longtemps été le point
    faible), ce qu'une F1 globale masque complètement."""
    by_pool: dict[str, list[float]] = {}
    for r in manifest.rows:
        if r["split"] != split:
            continue
        p = probas_by_file.get(r["file"])
        if p is not None:
            by_pool.setdefault(r["pool"], []).append(p)

    out = {}
    for pool, probas in sorted(by_pool.items()):
        arr = np.array(probas)
        fire = float((arr > threshold).mean())
        is_pos = any(r["pool"] == pool and r["label"] == 1 for r in manifest.rows)
        out[pool] = {
            "n": int(len(arr)),
            "mean_proba": float(arr.mean()),
            "fire_rate": fire,
            "expected": "haut" if is_pos else "bas",
            "ok": bool(fire > 0.85) if is_pos else bool(fire < 0.05),
        }
    return out


def confusion(y_true, y_proba, threshold: float = 0.5) -> np.ndarray:
    return confusion_matrix(np.asarray(y_true).astype(int),
                            (np.asarray(y_proba) > threshold).astype(int), labels=[0, 1])
