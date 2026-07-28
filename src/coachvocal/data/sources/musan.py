"""MUSAN noise — négatifs « bruit de fond ».

Il n'existe pas de corpus de bruit prêt-à-l'emploi en 1 s : la pratique standard
(Snips, openWakeWord, microWakeWord) est de cropper des enregistrements longs.
Le split est fait par POSITION dans le fichier source (0-70 % train, 70-85 %
val, 85-100 % test) pour éviter que deux crops voisins du même enregistrement
se retrouvent de part et d'autre de la frontière.

Trois gains (1.0 / 0.1 / 0.02) : le modèle doit rejeter le bruit fort ET le
bruit à peine audible, qui est ce qu'entend un micro always-on la plupart du temps.
"""

from __future__ import annotations

import random
from collections import Counter

import soundfile as sf

from ... import paths
from ...config import SourceConfig
from . import SourceContext, SplitPools, cached, mark_done, source


@source("musan_noise")
def musan_noise(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `budget`, `gains`, `subdir` (noise/music/speech)."""
    out_root = ctx.cache(src.name)
    if (hit := cached(out_root)) is not None:
        return hit

    noise_dir = paths.EXTERNAL / "musan" / src.params.get("subdir", "noise")
    if not noise_dir.exists():
        raise FileNotFoundError(f"MUSAN absent : {noise_dir} (cf. docs/DATA.md)")

    budget = dict(zip(("train", "val", "test"), src.params.get("budget", [1200, 150, 150])))
    gains = src.params.get("gains", [1.0, 0.1, 0.02])
    for s in budget:
        (out_root / s).mkdir(parents=True, exist_ok=True)

    files = sorted(noise_dir.rglob("*.wav"))
    random.Random(ctx.seed).shuffle(files)
    n = ctx.clip_samples
    counts: Counter = Counter()

    for file_idx, f in enumerate(files):
        if all(counts[s] >= budget[s] for s in budget):
            break
        try:
            audio, sr = sf.read(f, dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != ctx.sr or len(audio) < n:
            continue
        n_win = len(audio) // n
        for w in range(n_win):
            frac = w / max(n_win, 1)
            split = "train" if frac < 0.70 else ("val" if frac < 0.85 else "test")
            if counts[split] >= budget[split]:
                continue
            gain = gains[(w + file_idx) % len(gains)]
            sf.write(out_root / split / f"{f.stem}_{w:03d}_g{gain}.wav",
                     audio[w * n:(w + 1) * n] * gain, ctx.sr, subtype="PCM_16")
            counts[split] += 1

    print(f"    {src.name} : " + "  ".join(f"{s}:{counts[s]}" for s in budget))
    return mark_done(out_root)
