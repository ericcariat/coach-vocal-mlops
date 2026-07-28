"""Négatifs de **parole continue** découpés dans le corpus YouTube.

Déficit n°1 mis en évidence par le banc streaming : entraîné uniquement sur des
mots isolés (GSC, Common Voice tronqué à 1 s), le modèle n'a jamais vu ce qu'il
voit en production — un flux de parole ininterrompu où chaque fenêtre de 1 s
tombe au milieu d'un mot. D'où ~50 fausses alarmes/heure.

Deux garde-fous :
1. **Anti-fuite** : une fenêtre hérite du split de SA vidéo (`splits.csv`). Une
   vidéo du split test ne peut pas nourrir le train, et les vidéos du banc
   restent hors du train.
2. **Anti-contamination** : toute fenêtre qui recouvre une occurrence connue du
   mot-clé (± `guard_s`), ou une zone incertaine VTT, est écartée — on ne veut
   surtout pas étiqueter « négatif » une vraie prononciation.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter

import numpy as np
import soundfile as sf

from ...config import SourceConfig
from .. import corpus as corpus_mod
from . import SourceContext, SplitPools, cached, mark_done, source


def video_splits(ctx: SourceContext) -> dict[str, str]:
    """video_id → split, d'après les groupes `yt_<id>` de splits.csv."""
    out: dict[str, str] = {}
    csv_path = ctx.word_dir / ctx.dataset.splits_csv
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("source") != "youtube":
                continue
            m = re.match(rf"yt_({corpus_mod.VIDEO_ID})", row["group"])
            if m:
                out[m.group(1)] = row["split"]
    return out


@source("speech_negatives")
def speech_negatives(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `budget` (train, val, test), `stride_s`, `guard_s`, `min_peak`,
    `max_per_video`."""
    out_root = ctx.cache(src.name)
    if (hit := cached(out_root)) is not None:
        return hit

    budget = dict(zip(("train", "val", "test"), src.params.get("budget", [1500, 150, 150])))
    stride_s = float(src.params.get("stride_s", 1.0))
    guard_s = float(src.params.get("guard_s", 2.0))
    min_peak = float(src.params.get("min_peak", 0.02))
    max_per_video = int(src.params.get("max_per_video", 60))

    vid_split = video_splits(ctx)
    segments = corpus_mod.list_segments(ctx.wakeword.name, require_vtt=False)
    # Une vidéo inconnue de splits.csv n'a jamais servi : on la réserve au banc
    # streaming plutôt que de l'absorber dans le train.
    segments = [s for s in segments if s.video_id in vid_split]
    random.Random(ctx.seed).shuffle(segments)

    for s in budget:
        (out_root / s).mkdir(parents=True, exist_ok=True)

    n = ctx.clip_samples
    stride = int(stride_s * ctx.sr)
    counts: Counter = Counter()
    per_video: Counter = Counter()

    for seg in segments:
        split = vid_split[seg.video_id]
        if counts[split] >= budget.get(split, 0):
            continue
        try:
            audio, sr = sf.read(seg.wav, dtype="float32")
        except Exception:
            continue
        if sr != ctx.sr:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        forbidden = list(seg.occurrences or []) + list(seg.uncertain or [])
        for start in range(0, max(len(audio) - n, 0) + 1, stride):
            if counts[split] >= budget[split] or per_video[seg.video_id] >= max_per_video:
                break
            t0, t1 = start / ctx.sr, (start + n) / ctx.sr
            if any(t0 - guard_s <= o <= t1 + guard_s for o in forbidden):
                continue
            win = audio[start:start + n]
            if np.abs(win).max() < min_peak:      # fenêtre quasi muette → sans intérêt
                continue
            sf.write(out_root / split / f"{seg.wav.stem}_{start // ctx.sr:04d}.wav",
                     win, ctx.sr, subtype="PCM_16")
            counts[split] += 1
            per_video[seg.video_id] += 1

    print(f"    {src.name} : " + "  ".join(f"{s}:{counts[s]}" for s in budget)
          + f"  ({len(per_video)} vidéos)")
    if not sum(counts.values()):
        raise RuntimeError(
            f"{src.name} : aucune fenêtre produite — corpus vide ou splits.csv "
            "sans groupes `yt_<video_id>` ?")
    return mark_done(out_root)
