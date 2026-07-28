"""Clips réels du mot-clé et négatifs « proches », lus depuis `splits.csv`.

C'est la seule source dont le split n'est PAS calculé : il a été figé une fois
(par groupe = vidéo YouTube ou session d'enregistrement) et il est relu tel quel
à chaque run. Toute la comparabilité du CHANGELOG en dépend.
"""

from __future__ import annotations

from ...config import SourceConfig
from . import SourceContext, SplitPools, source


@source("word_clips")
def word_clips(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params :
    - `label_key` : clé dans splits.csv (`eloquence` pour les positifs, `proche`
      pour les négatifs phonétiquement voisins).
    - `prefix` : filtre optionnel sur le nom de fichier (`yt_` = YouTube,
      `moi_` = mes enregistrements). Permet de booster ma voix séparément.
    """
    label_key = src.params.get("label_key", ctx.wakeword.name)
    prefix = src.params.get("prefix")
    splits = ctx.word_splits()
    out = {}
    for s in ("train", "val", "test"):
        files = splits[s].get(label_key, [])
        out[s] = sorted(f for f in files if prefix is None or f.name.startswith(prefix))
    return out
