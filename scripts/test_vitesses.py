"""Test des vitesses — les clips de la voix de référence, accélérés/ralentis.

Chaque clip passe à l'examen final (mini-flux + machine à états réelle) en
version normale puis rééchantillonnée. Mesure directe du constat micro du
2026-08-14 : « il réagit quand j'appuie, pas aux mots rapides ».

    uv run python scripts/test_vitesses.py <modèle.onnx|model.keras> [...]
"""

from __future__ import annotations

import glob
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_oww_head import vitesse  # noqa: E402

from coachvocal import runtime  # noqa: E402
from coachvocal.config import load_wakeword  # noqa: E402
from coachvocal.inference.detector import load_detector  # noqa: E402

SR = 16000
FACTEURS = [(0.85, "×0.85"), (1.0, "normal"), (1.15, "×1.15"), (1.3, "×1.3")]


def souffle(n: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return 0.001 * rng.standard_normal(n).astype(np.float32)


def main():
    heads = [Path(a) for a in sys.argv[1:]]
    if not heads:
        sys.exit("usage : test_vitesses.py <modèle> [...]")
    runtime.configure(use_gpu=False)
    word = load_wakeword("eloquence")
    files = sorted(glob.glob(str(ROOT / "data/wakewords/eloquence/clean/positives/moi_*.wav")))

    print("Clips moi_ à plusieurs vitesses — déclenchement réel @0.8 :")
    print(f"{'modèle':34s}" + "".join(f"{lab:>9s}" for _, lab in FACTEURS))
    for head_path in heads:
        label = head_path.stem if head_path.stem != "model" else head_path.parent.name
        det = load_detector(head_path, word)
        ligne = f"{label[:33]:34s}"
        for f_sp, _lab in FACTEURS:
            fires = 0
            for p in files:
                a, sr = sf.read(p, dtype="float32")
                a = a.astype(np.float32)
                if f_sp != 1.0:
                    a = vitesse(a, f_sp)
                stream = np.concatenate([souffle(int(1.6 * SR), p + "|i"), a,
                                         souffle(int(0.6 * SR), p + "|o")])
                probas, peaks, _ = det.window_probas(stream)
                fires += bool(det.triggers_from(probas, peaks, 0.8))
            ligne += f"{fires / len(files):>9.0%}"
        print(ligne)


if __name__ == "__main__":
    main()
