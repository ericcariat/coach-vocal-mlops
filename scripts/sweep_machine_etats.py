"""Balayage de la MACHINE À ÉTATS du champion — aucun entraînement.

Le modèle n'a jamais changé de filtre de décision : fenêtres consécutives,
cooldown et seuil sont restés aux valeurs historiques. Ce balayage mesure,
pour chaque combinaison, le banc streaming (rappel, FA/h) ET la voix de
référence (mot entier, préfixes) — sans rien écrire dans les archives.

    uv run python scripts/sweep_machine_etats.py
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

from coachvocal import paths, registry, runtime  # noqa: E402
from coachvocal.config import load_wakeword  # noqa: E402
from coachvocal.data.sources.fragments import word_span  # noqa: E402
from coachvocal.evaluation import stream_bench  # noqa: E402
from coachvocal.inference.detector import load_detector  # noqa: E402

SR = 16000
SEUILS = (0.85, 0.9, 0.95)
COMBOS = [(nc, cd) for nc in (3, 4, 5) for cd in (1.0, 1.5)]


def souffle(n: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return 0.001 * rng.standard_normal(n).astype(np.float32)


def taux_voix(det, clips, seuil, prefixe_frac=None):
    """Part des clips qui DÉCLENCHENT (mini-flux + machine à états)."""
    feux = 0
    for chemin, audio in clips:
        a = audio
        if prefixe_frac is not None:
            w0, w1 = word_span(a, SR)
            a = a[w0:w0 + int(prefixe_frac * (w1 - w0))]
        flux = np.concatenate([souffle(int(1.6 * SR), chemin + "|i"), a,
                               souffle(int(0.6 * SR), chemin + "|o")])
        probas, peaks, _ = det.window_probas(flux)
        if det.triggers_from(probas, peaks, seuil):
            feux += 1
    return feux / len(clips)


def main():
    runtime.configure(use_gpu=False)
    modele = registry.model_path("eloquence")
    clips = []
    for chemin in sorted(glob.glob(str(ROOT / "data/wakewords/eloquence/clean/positives/moi_*.wav"))):
        audio, _ = sf.read(chemin, dtype="float32")
        clips.append((chemin, audio.astype(np.float32)))

    print(f"Champion : {registry.champion_run('eloquence')} — {len(COMBOS)} combinaisons\n")
    print(f"{'conséc.':>7s} {'cooldown':>8s} {'seuil':>6s} {'banc rappel':>11s} "
          f"{'FA/h':>6s} {'voix':>6s} {'préfixe90':>9s}")
    for nc, cd in COMBOS:
        word = load_wakeword("eloquence")
        word.live.n_consecutive = nc
        word.live.cooldown_s = cd
        payload = stream_bench.run({"champion": modele}, word, minutes=60,
                                   thresholds=SEUILS,
                                   splits_csv=paths.word_dir("eloquence") / "splits.csv")
        det = load_detector(modele, word)
        for seuil in SEUILS:
            r = payload["results"]["champion"][f"th{seuil}"]
            voix = taux_voix(det, clips, seuil)
            pref = taux_voix(det, clips, seuil, prefixe_frac=0.9)
            print(f"{nc:>7d} {cd:>8.1f} {seuil:>6.2f} "
                  f"{r['recall_stream']:>11.1%} {r['fa_per_hour']:>6.1f} "
                  f"{voix:>6.0%} {pref:>9.0%}", flush=True)


if __name__ == "__main__":
    main()
