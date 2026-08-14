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


def word_span(audio: np.ndarray, sr: int) -> tuple[int, int]:
    """Bornes du mot [début, fin) par énergie RELATIVE au pic du clip.

    Un seuil absolu (1e-5) est en-dessous du souffle d'un vrai micro
    (RMS ~1e-3) : il voyait « du mot » dès l'échantillon 0 et sur toute la
    seconde. Ici : RMS par trames de 20 ms, le mot = la zone au-dessus de
    10 % du RMS max — le souffle à ~1 % du pic vocal reste largement dessous.
    """
    frame = max(1, sr // 50)
    nf = max(1, len(audio) // frame)
    rms = np.sqrt((audio[: nf * frame].reshape(nf, frame) ** 2).mean(axis=1))
    idx = (rms >= 0.1 * float(rms.max())).nonzero()[0]
    if not len(idx):
        return 0, len(audio)
    return int(idx[0] * frame), int(min(len(audio), (idx[-1] + 1) * frame))


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
    # ⚠️ Défaut historique : la fraction porte sur la SECONDE du clip — pour un
    # mot de ~0,7 s, « 60 % du clip » peut contenir 80-100 % du mot, étiqueté
    # négatif (constaté à l'oreille par l'auteur le 2026-08-13 : des fragments qui
    # « passent pour le mot complet »). Le mode correctif `frac_of: word`
    # mesure le mot par énergie relative (`word_span`) et applique la fraction
    # AU MOT, découpé depuis SES bornes (pas celles du clip), plafonnée à
    # `max_word_frac` — un fragment reste un fragment. Opt-in : les recettes
    # historiques restent bit-à-bit intactes.
    # (Première version invalidée à l'oreille le 2026-08-14 : seuil absolu
    # 1e-5 < souffle micro, et découpe depuis la fin du clip — un f45 portait
    # 68 % du mot. D'où la mesure relative et l'ancrage sur les bornes.)
    frac_of = src.params.get("frac_of", "clip")
    max_word_frac = float(src.params.get("max_word_frac", 0.45))
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
            if frac_of == "word":
                w0, w1 = word_span(audio, ctx.sr)
            for frac in (fracs if is_dense else [fracs[i % len(fracs)]]):
                if frac_of == "word":
                    k = int(min(frac, max_word_frac) * (w1 - w0))
                    head, tail = audio[w0:w0 + k], audio[w1 - k:w1]
                else:
                    k = int(frac * n)
                    head, tail = audio[:k], audio[-k:]
                entering = np.concatenate([np.zeros(n - k, np.float32), head])
                leaving = np.concatenate([tail, np.zeros(n - k, np.float32)])
                tag = f"{path.stem}_f{int(frac * 100)}"
                sf.write(out_dir / f"{tag}_in.wav", entering, ctx.sr, subtype="PCM_16")
                sf.write(out_dir / f"{tag}_out.wav", leaving, ctx.sr, subtype="PCM_16")

    return _filter(mark_done(out_root))
