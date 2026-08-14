"""Examen par clips d'une tête openWakeWord — les MÊMES épreuves que les CNN.

Fait passer à une tête le test par clips des runs classiques : mêmes fichiers
(split test du manifest de référence), mêmes formules (`clip_eval`), mêmes
figures (matrice de confusion, compromis FRR/FAR, taux par pool). Écrit le
résultat dans le run de la tête (metrics.json + PNG) : la page Évaluation
l'affiche alors comme n'importe quel run.

Rappel ADR-004 : ces chiffres sont un signal de contrôle, pas un classement —
la promotion se joue au banc streaming.

    uv run python scripts/eval_oww_clips.py artifacts/runs/eloquence/<run>/model.onnx
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_oww_head import embed_files  # noqa: E402

from coachvocal.evaluation import clip_eval, report  # noqa: E402
from coachvocal.evaluation.oww_adapter import OWW_DIR  # noqa: E402

MANIFEST = ROOT / "artifacts/runs/eloquence/v17_stack/manifest.csv"
THRESHOLD = 0.5          # le seuil d'évaluation des runs CNN, pour comparer


def main():
    head_path = Path(sys.argv[1])
    run_dir = head_path.parent
    label = head_path.stem if head_path.stem != "model" else run_dir.name

    rows, seen = [], set()
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row["file"] not in seen and Path(row["file"]).exists():
                rows.append(row)
                seen.add(row["file"])
    files = [Path(r["file"]) for r in rows]
    y_true = np.array([int(r["label"]) for r in rows])
    print(f"📦  {len(files)} clips de test ({int(y_true.sum())} positifs) — {label}")

    import onnxruntime as ort
    mel = ort.InferenceSession(str(OWW_DIR / "melspectrogram.onnx"),
                               providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(str(OWW_DIR / "embedding_model.onnx"),
                               providers=["CPUExecutionProvider"])
    X = embed_files(files, mel, emb, "clipstest")
    head = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])
    name = head.get_inputs()[0].name
    proba = np.concatenate([head.run(None, {name: X[k:k + 1].astype(np.float32)})[0].ravel()
                            for k in range(len(X))])

    test = clip_eval.evaluate(y_true, proba, THRESHOLD)
    sweep = test.pop("threshold_sweep")
    print(f"    accuracy {test['accuracy']:.2%} · F1 {test['f1_pos']:.4f} · "
          f"FRR {test['frr']:.2%} · FAR {test['far']:.2%}")

    # Taux de déclenchement par pool (le diagnostic par famille)
    by_pool: dict[str, dict] = {}
    for r, p in zip(rows, proba):
        d = by_pool.setdefault(r["pool"], {"fires": 0, "n": 0, "pos": int(r["label"])})
        d["n"] += 1
        d["fires"] += int(p > THRESHOLD)
    breakdown = {pool: {
        "n": d["n"], "fire_rate": d["fires"] / d["n"],
        "expected": "haut" if d["pos"] else "bas",
        "ok": (d["fires"] / d["n"] > 0.85) if d["pos"] else (d["fires"] / d["n"] < 0.05),
    } for pool, d in by_pool.items()}

    report.confusion_png(clip_eval.confusion(y_true, proba, THRESHOLD),
                         ["négatif", "positif"], run_dir / "confusion.png",
                         title=f"{label} — test par clips @ {THRESHOLD}")
    report.threshold_png(sweep, run_dir / "threshold.png", live_threshold=0.8)
    report.pools_png(breakdown, run_dir / "pools.png")

    metrics_file = run_dir / "metrics.json"
    m = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    m["test"] = test
    m["pools_test"] = breakdown
    metrics_file.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"💾  {run_dir}/  (metrics.json, confusion.png, threshold.png, pools.png)")


if __name__ == "__main__":
    main()
