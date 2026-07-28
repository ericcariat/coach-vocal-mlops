"""Clips issus des sessions de test guidé (`coachvocal live guided`).

Ce sont des données en or : ma voix, au tempo réel, étiquetées à l'oreille juste
après la prononciation. Les TP/FN sont des positifs durs (le modèle les ratait),
les FP/TN sont des négatifs durs (le modèle se trompait dessus).

`exclude_prefixes` existe parce que huit clips d'une session enregistrée trop bas
(pic 0.06-0.11) tiraient le modèle vers un régime acoustique atypique une fois
boostés ×10 — cf. docs/JOURNAL.md (2026-07-21). À réessayer normalisés en gain.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from ...config import SourceConfig
from . import SourceContext, SplitPools, source

OUTCOMES_POS = ("TP", "FN")
OUTCOMES_NEG = ("FP", "TN")


def _center_crop(audio: np.ndarray, n: int, sr: int) -> np.ndarray:
    """Recadre 1 s autour du centre d'énergie (le mot n'est pas calé au début
    d'un enregistrement de 1.5 s déclenché par un décompte)."""
    win = sr // 100
    env = np.array([np.abs(audio[i * win:(i + 1) * win]).max() for i in range(len(audio) // win)])
    active = np.where(env > env.max() * 0.1)[0]
    center = int((active[0] + active[-1]) / 2) * win if len(active) else len(audio) // 2
    start = int(np.clip(center - n // 2, 0, max(len(audio) - n, 0)))
    crop = audio[start:start + n]
    return np.pad(crop, (0, max(0, n - len(crop))))


@source("guided")
def guided(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `outcomes` (défaut TP/FN si label=1, FP/TN sinon), `crop` (bool),
    `exclude_prefixes`, `dir` (défaut guided_clips)."""
    outcomes = tuple(src.params.get("outcomes",
                                    OUTCOMES_POS if src.label == 1 else OUTCOMES_NEG))
    exclude = tuple(src.params.get("exclude_prefixes", []))
    do_crop = src.params.get("crop", src.label == 1)

    src_dir = ctx.word_dir / src.params.get("dir", "guided_clips")
    out_dir = ctx.cache(src.name) / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = []
    if src_dir.exists():
        for f in sorted(src_dir.glob("*.wav")):
            if exclude and f.name.startswith(exclude):
                continue
            if f.stem.rsplit("_", 1)[-1] not in outcomes:
                continue
            if not do_crop:
                picked.append(f)
                continue
            out = out_dir / f.name
            if not out.exists():
                audio, _ = sf.read(f, dtype="float32")
                sf.write(out, _center_crop(audio, ctx.clip_samples, ctx.sr),
                         ctx.sr, subtype="PCM_16")
            picked.append(out)

    # Données de MA voix collectées après coup : elles ne peuvent servir qu'au
    # train, sinon le test cesserait d'être comparable d'un run à l'autre.
    return {"train": picked, "val": [], "test": []}
