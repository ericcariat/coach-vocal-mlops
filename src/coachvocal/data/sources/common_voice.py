"""Common Voice — négatifs « parole spontanée » (fr et en).

Deux précautions non négociables, héritées du run v01 :
1. On EXCLUT toute phrase dont la transcription contient le mot-clé — sinon on
   apprendrait littéralement au modèle que dire « éloquence » est un négatif.
2. Le split est fait par LOCUTEUR (hash md5 du `client_id`), pas par clip :
   deux phrases du même locuteur ne peuvent pas se retrouver de part et d'autre
   de la frontière train/test.
"""

from __future__ import annotations

import csv
import hashlib
import random
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from ... import paths
from ...config import SourceConfig
from . import SourceContext, SplitPools, cached, mark_done, source


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


@source("common_voice")
def common_voice(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `lang` (fr/en), `shard` (nom du .tar), `budget`, `offset_s`, `dir`."""
    lang = src.params.get("lang", "fr")
    out_root = ctx.cache(src.name)
    if (hit := cached(out_root)) is not None:
        return hit

    cv_dir = paths.EXTERNAL / src.params.get("dir", f"common_voice_{lang}")
    shard = src.params["shard"]
    budget = dict(zip(("train", "val", "test"), src.params.get("budget", [1200, 150, 150])))
    offset = src.params.get("offset_s", 0.5)   # saute le silence de tête fréquent
    stop_words = [strip_accents(w) for w in
                  src.params.get("exclude_containing", [ctx.wakeword.name[:7]])]

    shard_dir = cv_dir / shard
    if not shard_dir.exists():
        print(f"    extraction {shard}.tar …")
        subprocess.run(["tar", "-xf", str(cv_dir / f"{shard}.tar"), "-C", str(cv_dir)], check=True)
    available = {p.name for p in shard_dir.glob("*.mp3")}

    rows = []
    with open(cv_dir / "train.tsv", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sentence = strip_accents(row["sentence"])
            if row["path"] in available and not any(w in sentence for w in stop_words):
                rows.append((row["path"], row["client_id"]))
    random.Random(ctx.seed).shuffle(rows)

    for s in budget:
        (out_root / s).mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    margin = 1.3                                    # marge pour les échecs de décodage
    for path, client in rows:
        # hash déterministe : hash() de Python est randomisé à chaque processus
        h = int(hashlib.md5(client.encode()).hexdigest(), 16) % 10
        split = "train" if h < 8 else ("val" if h == 8 else "test")
        if counts[split] >= budget[split] * margin:
            if all(counts[s] >= budget[s] * margin for s in budget):
                break
            continue
        out = out_root / split / (Path(path).stem + ".wav")
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-ss", str(offset), "-t", str(ctx.wakeword.audio.clip_seconds),
             "-i", str(shard_dir / path), "-ar", str(ctx.sr), "-ac", "1", str(out)],
            capture_output=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 8000:
            counts[split] += 1
        else:
            out.unlink(missing_ok=True)

    print(f"    {src.name} : " + "  ".join(f"{s}:{counts[s]}" for s in budget))
    return mark_done(out_root)
