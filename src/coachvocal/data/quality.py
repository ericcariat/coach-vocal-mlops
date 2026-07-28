"""Audit qualité du dataset — « garbage in, garbage out ».

Contrôle systématique AVANT d'entraîner, parce qu'un run raté à cause des
données coûte plus cher qu'un run raté à cause du modèle, et qu'on ne le voit
pas dans la loss. Produit un dict JSON-able + un PNG (règle « preuves
visualisables » du projet).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .manifest import Manifest

# Pools dont la faible énergie est VOULUE : le pool `silence` est du bruit de
# plancher par construction, et les `fragments` sont complétés par des zéros.
# Les signaler comme anomalies noierait les vrais problèmes.
QUIET_BY_DESIGN = ("silence", "fragment")


def audit(manifest: Manifest, sample_rate: int, clip_seconds: float,
          max_per_pool: int = 300, seed: int = 42,
          quiet_by_design: tuple[str, ...] = QUIET_BY_DESIGN) -> dict:
    """Échantillonne chaque pool et mesure ce qui casse un entraînement audio :
    fichiers manquants, mauvaise fréquence, durée anormale, clip muet, saturation."""
    rng = np.random.default_rng(seed)
    pools: dict[str, list[Path]] = {}
    for r in manifest.rows:
        pools.setdefault(f"{r['split']}/{r['pool']}", []).append(Path(r["file"]))

    report: dict = {"pools": {}, "issues": []}
    for key, files in sorted(pools.items()):
        uniq = sorted({f for f in files})
        idx = rng.permutation(len(uniq))[:max_per_pool]
        peaks, durations, problems = [], [], Counter()
        for i in idx:
            f = uniq[i]
            if not f.exists():
                problems["missing"] += 1
                continue
            try:
                info = sf.info(f)
                audio, sr = sf.read(f, dtype="float32", frames=sample_rate * 4)
            except Exception:
                problems["unreadable"] += 1
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                problems["wrong_sr"] += 1
            durations.append(info.duration)
            peak = float(np.abs(audio).max()) if len(audio) else 0.0
            peaks.append(peak)
            if peak < 1e-3:
                problems["silent"] += 1
            if peak >= 0.999:
                problems["clipped"] += 1
            if abs(info.duration - clip_seconds) > 0.05:
                problems["duration"] += 1

        stats = {
            "n_files": len(uniq),
            "n_checked": int(len(idx)),
            "peak_median": float(np.median(peaks)) if peaks else 0.0,
            "peak_p10": float(np.percentile(peaks, 10)) if peaks else 0.0,
            "duration_median": float(np.median(durations)) if durations else 0.0,
            "problems": dict(problems),
        }
        report["pools"][key] = stats
        quiet_ok = any(tag in key for tag in quiet_by_design)
        for kind, n in problems.items():
            if kind == "silent" and quiet_ok:
                continue
            if kind in ("missing", "unreadable", "wrong_sr") or n > 0.1 * max(len(idx), 1):
                report["issues"].append(f"{key} : {n} clip(s) « {kind} »")

    report["fingerprint"] = manifest.fingerprint()
    report["ok"] = not report["issues"]
    return report


def plot(report: dict, out_png: Path, title: str = "Audit qualité du dataset") -> Path:
    """PNG : niveau sonore médian par pool + pools problématiques en évidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pools = report["pools"]
    keys = sorted(pools, key=lambda k: pools[k]["peak_median"])
    peaks = [pools[k]["peak_median"] for k in keys]
    flags = [bool(pools[k]["problems"]) for k in keys]
    colors = ["#C2452C" if f else "#5B6FB8" for f in flags]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(keys))))
    ax.barh(keys, peaks, color=colors)
    ax.set_xlabel("Amplitude crête médiane")
    ax.set_title(f"{title}\n(rouge = anomalies détectées)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png
