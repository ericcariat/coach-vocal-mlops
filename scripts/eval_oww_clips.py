"""Examen par clips en CONDITIONS RÉELLES — le clip entre progressivement.

Version finale de l'examen (2026-08-14) : chaque clip devient un mini-flux
(souffle léger, le clip, souffle léger — jamais de silence numérique) parcouru
par le VRAI détecteur : mêmes fenêtres glissantes, même portail d'énergie,
même règle des 3 fenêtres consécutives, même cooldown que le live et le banc
(`window_probas` + `triggers_from`). La question mesurée n'est plus « le
modèle peut-il reconnaître ce clip ? » mais « le détecteur AURAIT-IL SONNÉ ? »

Fonctionne pour les DEUX chaînes (le retour au CNN reste possible) :

    uv run python scripts/eval_oww_clips.py artifacts/runs/eloquence/<run>/model.onnx
    uv run python scripts/eval_oww_clips.py artifacts/runs/eloquence/<run>/model.keras

Écrit metrics.json (test), confusion.png, threshold.png, pools.png dans le
run. Rappel ADR-004 : signal de contrôle — la promotion se joue au banc.
"""

from __future__ import annotations

import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coachvocal import runtime  # noqa: E402
from coachvocal.config import load_wakeword  # noqa: E402
from coachvocal.evaluation import clip_eval, report  # noqa: E402
from coachvocal.inference.detector import load_detector  # noqa: E402

MANIFEST = ROOT / "artifacts/runs/eloquence/v17_stack/manifest.csv"
THRESHOLD = 0.5                    # seuil de référence des runs CNN
GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SR = 16000
LEAD_S, TAIL_S = 1.6, 0.6          # souffle avant/après (le mot ENTRE, puis sort)
AMP = 0.001                        # souffle léger de micro — jamais de zéros


def souffle(n: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return AMP * rng.standard_normal(n).astype(np.float32)


def main():
    model_path = Path(sys.argv[1])
    run_dir = model_path.parent
    label = model_path.stem if model_path.stem != "model" else run_dir.name
    if not (run_dir / "config.json").exists():   # candidat hors registre :
        run_dir = run_dir / label                # ses sorties vont dans un
        run_dir.mkdir(exist_ok=True)             # sous-dossier à son nom

    rows, seen = [], set()
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row["file"] not in seen and Path(row["file"]).exists():
                rows.append(row)
                seen.add(row["file"])
    y_true = np.array([int(r["label"]) for r in rows])
    print(f"📦  {len(rows)} clips de test ({int(y_true.sum())} positifs) — {label} "
          f"(machine à états réelle)")

    runtime.configure(use_gpu=False)              # ADR-002
    word = load_wakeword("eloquence")
    det = load_detector(model_path, word)

    # Pseudo-score par clip = le seuil le plus HAUT auquel le détecteur sonne
    # encore (monotone : sonner à 0.8 implique sonner à 0.5). 0 = jamais.
    pseudo = np.zeros(len(rows), np.float32)
    for i, r in enumerate(rows):
        a, sr = sf.read(r["file"], dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        stream = np.concatenate([souffle(int(LEAD_S * SR), r["file"] + "|in"),
                                 a.astype(np.float32),
                                 souffle(int(TAIL_S * SR), r["file"] + "|out")])
        probas, peaks, _ = det.window_probas(stream)
        for th in GRID:
            if det.triggers_from(probas, peaks, th):
                pseudo[i] = th + 0.005        # « sonne encore à th »
            else:
                break
        if i % 200 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    test = clip_eval.evaluate(y_true, pseudo, THRESHOLD)
    sweep = test.pop("threshold_sweep")
    print(f"    accuracy {test['accuracy']:.2%} · F1 {test['f1_pos']:.4f} · "
          f"FRR {test['frr']:.2%} · FAR {test['far']:.2%}  (déclenchement réel @ {THRESHOLD})")

    by_pool: dict[str, dict] = {}
    for r, p in zip(rows, pseudo):
        d = by_pool.setdefault(r["pool"], {"fires": 0, "n": 0, "pos": int(r["label"])})
        d["n"] += 1
        d["fires"] += int(p > THRESHOLD)
    breakdown = {pool: {
        "n": d["n"], "fire_rate": d["fires"] / d["n"],
        "expected": "haut" if d["pos"] else "bas",
        "ok": (d["fires"] / d["n"] > 0.85) if d["pos"] else (d["fires"] / d["n"] < 0.05),
    } for pool, d in by_pool.items()}

    report.confusion_png(clip_eval.confusion(y_true, pseudo, THRESHOLD),
                         ["négatif", "positif"], run_dir / "confusion.png",
                         title=f"{label} — déclenchements réels @ {THRESHOLD}")
    report.threshold_png(sweep, run_dir / "threshold.png", live_threshold=0.8)
    report.pools_png(breakdown, run_dir / "pools.png")

    metrics_file = run_dir / "metrics.json"
    m = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    m["test"] = test
    m["test_protocole"] = ("mini-flux (souffle + clip + souffle), machine à "
                           "états réelle — le clip entre progressivement")
    m["pools_test"] = breakdown
    metrics_file.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"💾  {run_dir}/  (metrics.json, confusion.png, threshold.png, pools.png)")


if __name__ == "__main__":
    main()
