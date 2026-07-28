"""Fragments entrant / sortant — négatifs contre le déclenchement prématuré.

En streaming, la fenêtre de 1 s voit d'abord le DÉBUT du mot (« élo… »), puis le
mot entier, puis la FIN (« …quence »). Sans ces négatifs, le modèle déclenche dès
« élo » et rate le vrai centrage, ce qui dégrade l'alignement des triggers.

Génération PAR SPLIT : le fragment d'un clip de test ne peut pas atterrir dans le
train — sinon on entraîne indirectement sur le test.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from ...config import SourceConfig
from . import SourceContext, SplitPools, cached, mark_done, source


@source("fragments")
def fragments(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `prefix_fracs` (fraction du mot visible), `subset` (1 clip yt sur N),
    `dense_prefix` (préfixe des clips traités avec toutes les fractions)."""
    # Le pool est généré une fois puis filtré : `moi_` et `yt_` sont deux
    # sources distinctes dans la recette (boosts différents), mais un seul cache.
    out_root = ctx.cache(src.params.get("pool", "fragments"))
    prefix = src.params.get("prefix")

    def _filter(pools: SplitPools) -> SplitPools:
        if not prefix:
            return pools
        return {s: [f for f in files if f.name.startswith(prefix)] for s, files in pools.items()}

    if (hit := cached(out_root)) is not None:
        return _filter(hit)

    fracs = src.params.get("prefix_fracs", [0.30, 0.45, 0.60])
    subset = src.params.get("subset", 3)
    dense_prefix = src.params.get("dense_prefix", "moi_")
    n = ctx.clip_samples
    splits = ctx.word_splits()

    for split in ("train", "val", "test"):
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(sorted(splits[split].get(ctx.wakeword.name, []))):
            is_dense = path.name.startswith(dense_prefix)
            if not is_dense and i % subset != 0:
                continue
            audio, _ = sf.read(path, dtype="float32")
            for frac in (fracs if is_dense else [fracs[i % len(fracs)]]):
                k = int(frac * n)
                entering = np.concatenate([np.zeros(n - k, np.float32), audio[:k]])
                leaving = np.concatenate([audio[-k:], np.zeros(n - k, np.float32)])
                tag = f"{path.stem}_f{int(frac * 100)}"
                sf.write(out_dir / f"{tag}_in.wav", entering, ctx.sr, subtype="PCM_16")
                sf.write(out_dir / f"{tag}_out.wav", leaving, ctx.sr, subtype="PCM_16")

    return _filter(mark_done(out_root))
