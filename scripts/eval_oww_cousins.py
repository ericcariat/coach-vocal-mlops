"""Banc des cousins — la voix de référence, mesurée en conditions réelles.

Chaque clip devient un mini-flux (souffle léger, le clip, souffle léger) que
parcourt le VRAI détecteur : fenêtres glissantes, portail d'énergie, 3
fenêtres consécutives, cooldown — le même code que le live et le banc. On
compte les DÉCLENCHEMENTS, pas les pics isolés (examen finalisé 2026-08-14).

Familles mesurées :
  - positifs moi_   : le mot entier — doit sonner        [vus à l'entraînement]
  - cousins moi_    : « éloquente », « élégance »… — muets [vus si adv-weight]
  - cousins TTS     : jamais vus — muets
  - hard negatives  : fausses alarmes du banc — muets     [vus]
  - préfixes 80/90 %: le mot coupé avant la fin — muets

Fonctionne pour les deux chaînes (ONNX ou keras). Sortie : taux par seuil,
JSON fusionné pour la page Évaluation (section Ma voix), PNG des probas max.

    uv run python scripts/eval_oww_cousins.py <modèle.onnx|model.keras> [...]
"""

from __future__ import annotations

import datetime
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
from coachvocal.data.sources.fragments import word_span  # noqa: E402
from coachvocal.inference.detector import load_detector  # noqa: E402

SR = 16000
THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.99]
LEAD_S, TAIL_S = 1.6, 0.6
AMP = 0.001                        # souffle léger — jamais de silence numérique


def souffle(n: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return AMP * rng.standard_normal(n).astype(np.float32)


def charge(f: Path) -> np.ndarray | None:
    a, sr = sf.read(f, dtype="float32")
    if sr != SR:
        return None
    return a.mean(axis=1).astype(np.float32) if a.ndim > 1 else a.astype(np.float32)


def groupes() -> list[tuple[str, list[np.ndarray], bool]]:
    out = []
    pos = [charge(f) for f in sorted(
        (ROOT / "data/wakewords/eloquence/clean/positives").glob("moi_*.wav"))]
    pos = [a for a in pos if a is not None]
    out.append(("positifs moi_ (attendu : HAUT, vus)", pos, True))
    for nom, pattern, base in [
        ("cousins moi_ (attendu : BAS, vus si adv-weight)", "moi_*.wav",
         ROOT / "data/wakewords/eloquence/clean/negatives_proches"),
        ("cousins TTS (attendu : BAS, jamais vus)", "*.wav",
         ROOT / "data/wakewords/eloquence/generated/tts_neg_proches"),
        ("hard negatives banc (attendu : BAS, vus)", "hn_*.wav",
         ROOT / "exports/oww_training_b/negatifs_adversariaux"),
    ]:
        arrs = [charge(f) for f in sorted(base.glob(pattern))[:150]]
        out.append((nom, [a for a in arrs if a is not None], False))
    for frac in (0.8, 0.9):
        arrs = []
        for a in pos:
            w0, w1 = word_span(a, SR)
            arrs.append(a[w0:w0 + int(frac * (w1 - w0))])
        out.append((f"préfixes {frac:.0%} du mot (attendu : BAS)", arrs, False))
    return out


def main():
    heads = [Path(a) for a in sys.argv[1:]]
    if not heads:
        sys.exit("usage : eval_oww_cousins.py <modèle.onnx|model.keras> [...]")

    runtime.configure(use_gpu=False)              # ADR-002
    word = load_wakeword("eloquence")
    familles = groupes()

    json_out = ROOT / "artifacts/reports/oww_cousins.json"
    store = json.loads(json_out.read_text()) if json_out.exists() else {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(heads), 1, figsize=(9, 3.2 * len(heads)),
                             squeeze=False)
    for hi, head_path in enumerate(heads):
        label = head_path.stem if head_path.stem != "model" else head_path.parent.name
        det = load_detector(head_path, word)
        print(f"\n=== {label} (machine à états réelle) ===")
        ax = axes[hi][0]
        entry = {"date": datetime.date.today().isoformat(),
                 "protocole": "mini-flux + machine à états", "groupes": {}}
        for nom, arrs, attendu_haut in familles:
            fires = {th: 0 for th in THRESHOLDS}
            pmax = []
            for k, a in enumerate(arrs):
                stream = np.concatenate([souffle(int(LEAD_S * SR), f"{nom}|{k}|in"),
                                         a, souffle(int(TAIL_S * SR), f"{nom}|{k}|out")])
                probas, peaks, _ = det.window_probas(stream)
                pmax.append(float(probas.max()) if len(probas) else 0.0)
                for th in THRESHOLDS:
                    if det.triggers_from(probas, peaks, th):
                        fires[th] += 1
            n = len(arrs)
            taux = " · ".join(f"@{th}: {fires[th] / n:.0%}" for th in THRESHOLDS)
            print(f"  {nom} ({n}) — {taux}")
            entry["groupes"][nom] = {
                "n": n, "attendu": "haut" if attendu_haut else "bas",
                "taux": {str(th): round(fires[th] / n, 4) for th in THRESHOLDS}}
            ax.hist(pmax, bins=40, range=(0, 1), alpha=0.55, label=f"{nom} (n={n})")
        store[label] = entry
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("probabilité max sur le mini-flux")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = ROOT / "artifacts/reports/oww_cousins.png"
    fig.savefig(out, dpi=120)
    json_out.write_text(json.dumps(store, indent=1, ensure_ascii=False))
    print(f"\n💾  {out}\n💾  {json_out}")


if __name__ == "__main__":
    main()
