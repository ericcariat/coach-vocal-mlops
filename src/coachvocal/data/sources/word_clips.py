"""Clips réels du mot-clé et négatifs « proches », lus depuis `splits.csv`.

C'est la seule source dont le split n'est PAS calculé : il a été figé une fois
(par groupe = vidéo YouTube ou session d'enregistrement) et il est relu tel quel
à chaque run. Toute la comparabilité du CHANGELOG en dépend.
"""

from __future__ import annotations

import random

import numpy as np
import soundfile as sf

from ...config import SourceConfig
from . import SourceContext, SplitPools, cached, mark_done, source


@source("word_clips")
def word_clips(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params :
    - `label_key` : clé dans splits.csv (`eloquence` pour les positifs, `proche`
      pour les négatifs phonétiquement voisins).
    - `prefix` : filtre optionnel sur le nom de fichier (`yt_` = YouTube,
      `moi_` = mes enregistrements). Permet de booster ma voix séparément.
    - `re_anchor` (opt-in) : ré-ancre le mot dans la fenêtre du TRAIN — fin du
      mot à une marge aléatoire [0, `jitter_s`] du bord droit, mot jamais
      tronqué. Mesuré : 32/47 clips `moi_` ont le mot collé à la fin du clip,
      un `time_shift` de +100 ms en coupait la fin (2026-08-15). Les recettes
      qui l'activent mettent `time_shift_ms: 0`. Val/test restent intacts
      (mêmes fichiers que toutes les versions précédentes).
    """
    label_key = src.params.get("label_key", ctx.wakeword.name)
    prefix = src.params.get("prefix")
    splits = ctx.word_splits()
    out = {}
    for s in ("train", "val", "test"):
        files = splits[s].get(label_key, [])
        out[s] = sorted(f for f in files if prefix is None or f.name.startswith(prefix))

    if not src.params.get("re_anchor"):
        return out

    from .fragments import word_span

    jitter_s = float(src.params.get("jitter_s", 0.2))
    pool = src.params.get("pool", f"anchored_{label_key}_{prefix or 'tous'}")
    out_root = ctx.cache(pool)
    if (hit := cached(out_root)) is not None:
        return {**out, "train": hit["train"]}

    n = ctx.clip_samples
    for s in ("train", "val", "test"):        # val/test : dossiers vides (contrat cache)
        (out_root / s).mkdir(parents=True, exist_ok=True)
    for path in out["train"]:
        audio, _ = sf.read(path, dtype="float32")
        audio = audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))
        w0, w1 = word_span(audio, ctx.sr)
        rng = random.Random(f"{ctx.seed}|anchor|{path.name}")
        margin = int(rng.uniform(0.0, jitter_s) * ctx.sr)
        margin = min(margin, n - (w1 - w0))   # le mot entier tient toujours
        shift = (n - margin) - w1             # amène la fin du mot à n - marge
        moved = np.zeros(n, np.float32)
        src_lo, src_hi = max(0, -shift), min(n, n - shift)
        moved[src_lo + shift:src_hi + shift] = audio[src_lo:src_hi]
        sf.write(out_root / "train" / path.name, moved, ctx.sr, subtype="PCM_16")
    return {**out, "train": mark_done(out_root)["train"]}
