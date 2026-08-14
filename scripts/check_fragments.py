"""Contrôle du pool fragments_word : quelle fraction du mot chaque fragment porte-t-il ?

Vérification INDÉPENDANTE de la génération (idée de l'auteur, 2026-08-14, à la place
d'un repassage WhisperX) : on remesure à l'énergie la voix contenue dans chaque
fragment et on la rapporte à la durée du mot dans son clip source. Sortie :
un histogramme PNG dans `artifacts/reports/` + code retour 1 si un fragment
dépasse le plafond (45 % + 5 points de tolérance de mesure).

    uv run python scripts/check_fragments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coachvocal.data.sources.fragments import word_span  # noqa: E402

POOL = Path("artifacts/cache/eloquence/fragments_word")
CLEAN = Path("data/wakewords/eloquence/clean/positives")
PLAFOND = 0.45 + 0.05

fractions, fautifs = [], []
for frag_path in sorted(POOL.glob("*/*.wav")):
    audio, sr = sf.read(frag_path, dtype="float32")
    f0, f1 = word_span(audio, sr)
    stem = frag_path.stem.rsplit("_f", 1)[0]
    src = CLEAN / f"{stem}.wav"
    if not src.exists():
        continue
    clip, csr = sf.read(src, dtype="float32")
    w0, w1 = word_span(clip, csr)
    frac = (f1 - f0) / sr / max(1e-6, (w1 - w0) / csr)
    fractions.append(frac)
    if frac > PLAFOND:
        fautifs.append((frag_path, frac))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(fractions, bins=30, color="#2a78d6", edgecolor="white")
ax.axvline(0.45, color="#eb6834", lw=2, label="plafond max_word_frac = 45 %")
ax.set_xlabel("Fraction du mot embarquée dans le fragment (mesure énergie)")
ax.set_ylabel("Nombre de fragments")
ax.set_title(f"Pool fragments_word — {len(fractions)} fragments contrôlés, "
             f"{len(fautifs)} au-dessus du plafond")
ax.legend()
fig.tight_layout()
out = Path("artifacts/reports/fragments_word_controle.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)

print(f"{len(fractions)} fragments contrôlés — médiane "
      f"{np.median(fractions):.0%}, max {max(fractions):.0%} → {out}")
for p, f in fautifs[:10]:
    print(f"  DÉPASSEMENT {f:.0%} : {p}")
sys.exit(1 if fautifs else 0)
