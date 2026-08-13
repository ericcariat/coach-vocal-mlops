"""Clips du STUDIO d'enregistrement guidé (page Streamlit « Studio »).

Campagnes scriptées à la ViolaWake (ROADMAP P2) : N prises par condition
(normal, fort, joyeux, rapide, lent, 50 cm, 1 m, 2,5 m…), contrôle qualité
immédiat à la prise (les mêmes mesures que la porte qualité, ADR-007), et
**une session = un groupe indivisible** : comme les clips guidés, les prises
studio ne vont QU'AU TRAIN — le test resterait sinon incomparable d'un run à
l'autre.

Arborescence : `data/wakewords/<mot>/studio/<session>/<condition>_<nn>.wav`
+ `metadata.json` par session (mesures, verdicts « garder/refaire », matériel).
Seules les prises `keep: true` du metadata sont utilisées.
"""

from __future__ import annotations

import json

import soundfile as sf

from ...config import SourceConfig
from . import SourceContext, SplitPools, source
from .guided import _center_crop


@source("studio")
def studio(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `exclude_sessions` (liste de noms), `crop` (défaut true),
    `align` (« center » historique ou « end » : fin du mot à 0-`jitter_s` du
    bord droit — géométrie du déclenchement streaming), `jitter_s` (0.2),
    `dir` (défaut studio)."""
    import zlib

    import numpy as np

    from ..tts import place_word

    exclude = set(src.params.get("exclude_sessions", []))
    do_crop = bool(src.params.get("crop", True))
    align = str(src.params.get("align", "center"))
    jitter_s = float(src.params.get("jitter_s", 0.2))
    root = ctx.word_dir / src.params.get("dir", "studio")
    out_root = ctx.cache(src.name) / "train"
    out_root.mkdir(parents=True, exist_ok=True)

    picked = []
    if root.exists():
        for session in sorted(p for p in root.iterdir() if p.is_dir()):
            if session.name in exclude:
                continue
            meta_path = session / "metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            takes = meta.get("takes", {})
            for f in sorted(session.glob("*.wav")):
                if not takes.get(f.name, {}).get("keep", False):
                    continue                     # jamais de prise non validée
                if not do_crop:
                    picked.append(f)
                    continue
                out = out_root / f"{session.name}_{f.name}"
                if not out.exists():
                    audio, _ = sf.read(f, dtype="float32")
                    if align == "end":
                        # rogner les silences puis caler la fin du mot au bord
                        env = np.abs(audio)
                        active = (env > env.max() * 0.1).nonzero()[0]
                        word = audio[active[0]:active[-1] + 1] if len(active) else audio
                        rng = np.random.default_rng(zlib.crc32(out.name.encode()))
                        margin = int(rng.uniform(0, jitter_s) * ctx.sr)
                        clip = place_word(word, ctx.clip_samples, "end", margin)
                    else:
                        clip = _center_crop(audio, ctx.clip_samples, ctx.sr)
                    sf.write(out, clip, ctx.sr, subtype="PCM_16")
                picked.append(out)

    # Même règle que les clips guidés : ma voix collectée après coup → train
    # uniquement, sinon le test cesse d'être comparable entre runs.
    return {"train": picked, "val": [], "test": []}
