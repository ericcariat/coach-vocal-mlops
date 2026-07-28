"""Positifs synthétiques Piper (TTS) — augmentation de la diversité de voix.

Enseignement du sweep dose-réponse (docs/JOURNAL.md, 2026-07-21) : le gain est
réel mais **non monotone**. À 500 clips (~2× le nombre de positifs réels) la F1
monte et la FAR baisse sur 3 seeds appariés ; à 2000 le synthétique noie le réel
et tout s'effondre. La dose est donc un hyperparamètre de la recette, pas une
constante — d'où `dose` dans le YAML.

L'échantillonnage est **stratifié par combinaison** (voix, locuteur, vitesse,
bruit de sampling) : à dose fixée, on garde la même couverture de l'espace des
voix, et deux runs à la même seed tirent exactement les mêmes fichiers.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from ...config import SourceConfig
from . import SourceContext, SplitPools, source

STRATA = ("voice", "speaker", "length_scale", "noise_scale")


def stratified_sample(rows: list[dict], n: int, seed: int,
                      strata: tuple[str, ...] = STRATA) -> list[dict]:
    """Tirage en tourniquet sur les strates : on prend le 1er de chaque combo,
    puis le 2e, etc. jusqu'à `n`. Sous-ensemble emboîté : dose 100 ⊂ dose 500."""
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[tuple(r.get(k, "") for k in strata)].append(r)
    rng = random.Random(seed)
    for combo in by_combo.values():
        rng.shuffle(combo)
    picked: list[dict] = []
    i = 0
    while len(picked) < min(n, len(rows)):
        for combo in sorted(by_combo):
            if i < len(by_combo[combo]) and len(picked) < n:
                picked.append(by_combo[combo][i])
        i += 1
    return picked


@source("tts_piper")
def tts_piper(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `pool` (dossier sous generated/), `dose` (nb de clips)."""
    pool_dir = ctx.word_dir / "generated" / src.params.get("pool", "tts_positives")
    manifest = pool_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"pool TTS absent : {manifest}\n"
            "    → générer d'abord : coachvocal data tts-pool --wakeword "
            f"{ctx.wakeword.name}"
        )
    rows = list(csv.DictReader(manifest.open(newline="")))
    dose = int(src.params.get("dose", len(rows)))
    picked = stratified_sample(rows, dose, ctx.seed)
    files = [Path(pool_dir / r["file"]) for r in picked]

    # Le synthétique n'entre JAMAIS en val/test : sinon on mesurerait la capacité
    # à reconnaître Piper, pas à reconnaître une voix humaine.
    return {"train": files, "val": [], "test": []}
