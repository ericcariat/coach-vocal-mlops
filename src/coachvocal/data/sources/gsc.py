"""Google Speech Commands v2 — négatifs « mots isolés d'une autre langue ».

On réutilise les splits OFFICIELS du corpus (`validation_list.txt` /
`testing_list.txt`) : ils sont construits par hash du locuteur, donc une même
voix ne traverse jamais la frontière train/test. Les chiffres sont exclus (ils
servaient au chantier marvin et n'apportent rien ici).
"""

from __future__ import annotations

import random

from ... import paths
from ...config import SourceConfig
from . import SourceContext, SplitPools, source

DIGIT_WORDS = {"zero", "one", "two", "three", "four", "five",
               "six", "seven", "eight", "nine"}


@source("gsc")
def gsc(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `budget` (train, val, test), `root` (défaut data/external/gsc)."""
    root = paths.EXTERNAL / src.params.get("root", "gsc") / "raw"
    if not root.exists():
        raise FileNotFoundError(f"GSC absent : {root} (cf. docs/DATA.md pour le téléchargement)")

    val_set = set((root / "validation_list.txt").read_text().splitlines())
    test_set = set((root / "testing_list.txt").read_text().splitlines())
    exclude = set(src.params.get("exclude_words", [])) | DIGIT_WORDS | {"_background_noise_"}

    words = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in exclude)
    pool: SplitPools = {s: [] for s in ("train", "val", "test")}
    for word in words:
        for f in sorted((root / word).glob("*.wav")):
            rel = f"{word}/{f.name}"
            split = "val" if rel in val_set else ("test" if rel in test_set else "train")
            pool[split].append(f)

    budget = dict(zip(("train", "val", "test"), src.params.get("budget", [2000, 250, 250])))
    rng = random.Random(ctx.seed)
    for s in pool:
        rng.shuffle(pool[s])
        pool[s] = sorted(pool[s][: budget[s]])
    return pool
