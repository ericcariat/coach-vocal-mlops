"""Négatifs de parole continue depuis un DOSSIER de longs enregistrements.

Complément de `speech_negatives` (qui découpe le corpus YouTube) : ici la
source est un dossier de WAV longs, 16 kHz mono, **certifiés sans le mot-clé**
(transcriptions vérifiées à l'import — cf. docs/DATA.md). Premier usage : les
réunions du split TRAIN de SUMM-RE — le domaine exact où le banc mesure le
plus de fausses alarmes (~72/h sur les réunions contre ~48/h sur YouTube).

Anti-fuite : ces fichiers viennent de réunions DISJOINTES de celles du banc
(004c/006b/015b) et de val_ambient (017a/018a) — la disjonction est vérifiée à
l'import et documentée dans DATA.md. Tout va au train uniquement.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from ...config import SourceConfig
from . import SourceContext, SplitPools, cached, mark_done, source


@source("continuous_dir")
def continuous_dir(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `dir` (relatif à la racine du projet ou absolu), `budget`
    (nb de fenêtres, train uniquement), `stride_s`, `min_peak`,
    `max_per_file`."""
    out_root = ctx.cache(src.name)
    if (hit := cached(out_root)) is not None:
        return hit

    from ... import paths as project_paths
    raw = str(src.params["dir"])
    directory = Path(raw) if raw.startswith("/") else project_paths.ROOT / raw
    budget = int(src.params.get("budget", 300))
    stride_s = float(src.params.get("stride_s", 1.0))
    min_peak = float(src.params.get("min_peak", 0.02))
    max_per_file = int(src.params.get("max_per_file", 200))

    wavs = sorted(directory.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"{src.name} : aucun wav dans {directory} — "
                           "importer d'abord (cf. docs/DATA.md)")
    rng = random.Random(ctx.seed)
    rng.shuffle(wavs)

    (out_root / "train").mkdir(parents=True, exist_ok=True)
    n = ctx.clip_samples
    stride = int(stride_s * ctx.sr)
    counts: Counter = Counter()

    for wav in wavs:
        if counts["train"] >= budget:
            break
        try:
            audio, sr = sf.read(wav, dtype="float32")
        except Exception:
            continue
        if sr != ctx.sr:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Tirage de départs aléatoires (déterministe) plutôt que balayage
        # séquentiel : un budget modeste échantillonne tout l'enregistrement.
        starts = list(range(0, max(len(audio) - n, 0) + 1, stride))
        rng.shuffle(starts)
        taken = 0
        for start in starts:
            if counts["train"] >= budget or taken >= max_per_file:
                break
            win = audio[start:start + n]
            if np.abs(win).max() < min_peak:
                continue
            sf.write(out_root / "train" / f"{wav.stem}_{start // ctx.sr:05d}.wav",
                     win, ctx.sr, subtype="PCM_16")
            counts["train"] += 1
            taken += 1

    print(f"    {src.name} : train:{counts['train']}  ({len(wavs)} fichier(s) source)")
    if not counts["train"]:
        raise RuntimeError(f"{src.name} : aucune fenêtre produite")
    return mark_done(out_root)
