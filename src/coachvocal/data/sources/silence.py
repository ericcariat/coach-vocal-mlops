"""Quasi-silence — bruit blanc de très faible amplitude.

Sans cette source, un détecteur always-on part en vrille dans une pièce vide :
la normalisation z-score amplifie le bruit de plancher du micro et le
spectrogramme obtenu ressemble à n'importe quoi. Ces clips sont générés (pas
téléchargés) donc parfaitement reproductibles depuis la seed.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from ...config import SourceConfig
from . import SourceContext, SplitPools, cached, mark_done, source


@source("silence")
def silence(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `budget`, `amp_min`, `amp_max`."""
    out_root = ctx.cache(src.name)
    if (hit := cached(out_root)) is not None:
        return hit

    budget = src.params.get("budget", [150, 20, 20])
    amp_min = src.params.get("amp_min", 1e-5)
    amp_max = src.params.get("amp_max", 3e-4)
    rng = np.random.default_rng(ctx.seed)

    for split, count in zip(("train", "val", "test"), budget):
        (out_root / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            amp = rng.uniform(amp_min, amp_max)
            clip = rng.normal(0.0, amp, ctx.clip_samples).astype(np.float32)
            # Le split fait partie du nom : le contrôle anti-fuite compare les
            # noms de fichiers, et des homonymes entre splits le déclencheraient
            # à tort (ces clips sont distincts, seul leur numéro se répète).
            sf.write(out_root / split / f"silence_{split}_{i:04d}.wav", clip,
                     ctx.sr, subtype="PCM_16")

    print(f"    {src.name} : " + "  ".join(f"{s}:{n}" for s, n in
                                           zip(("train", "val", "test"), budget)))
    return mark_done(out_root)
