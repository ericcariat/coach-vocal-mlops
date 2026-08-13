"""Positifs YouTube RE-DÉCOUPÉS : fin du mot près de la fin de fenêtre.

Pourquoi (audit + littérature du 2026-08-13, cf. `docs/ROADMAP.md` et
`docs/FENETRE_GLISSANTE_ET_JITTER.html`) : les clips actuels calent le mot à
l'échantillon 0 et remplissent la fin de zéros — l'inverse exact de la
géométrie rencontrée en streaming, où la fenêtre qui déclenche contient du
contexte réel PUIS le mot, dont la fin touche presque le bord droit. Le padding
de silence après le mot est en outre un indice que le modèle peut apprendre et
qui n'existe jamais en production.

Recette de découpe, par occurrence :
- fin de fenêtre = `t_end` du mot + marge tirée dans [0, `jitter_s`] (déterministe
  par occurrence via la seed données) ;
- début de fenêtre = fin − `clip_seconds`, rempli de CONTEXTE RÉEL (la parole
  qui précède dans la vidéo) ;
- même vidéo → même split (`splits.csv`, anti-fuite identique aux autres pools).

⚠️ Deux interactions à connaître :
- `augmentation.time_shift_ms` doit être ≈ 0 dans les recettes qui utilisent ce
  pool : le jitter est déjà dans la découpe, et un shift positif pousserait la
  fin du mot HORS de la fenêtre.
- Le contrôle « fin chargée » de la porte qualité (ADR-007) ne s'applique pas à
  cette géométrie (l'énergie en fin de fenêtre est voulue) : retirer ce pool de
  `quality_gate.tail_check_pools` dans la recette.
"""

from __future__ import annotations

import random
from collections import Counter

import soundfile as sf

from ...config import SourceConfig
from .. import corpus as corpus_mod
from . import SourceContext, SplitPools, cached, mark_done, source
from .speech_negatives import video_splits


@source("word_clips_recut")
def word_clips_recut(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params :
    - `jitter_s` (défaut 0.2) : marge max entre fin du mot et fin de fenêtre.
    - `only_known_positives` (défaut true) : ne re-découpe que les occurrences
      correspondant à un clip positif de `splits.csv` (même variable, seule la
      découpe change) ; à false, toutes les occurrences des vidéos des splits.
    """
    out_root = ctx.cache(src.name)
    if (hit := cached(out_root)) is not None:
        return hit

    jitter_s = float(src.params.get("jitter_s", 0.2))
    only_known = bool(src.params.get("only_known_positives", True))
    clip_s = ctx.wakeword.audio.clip_seconds
    n = ctx.clip_samples

    vid_split = video_splits(ctx)
    spans = corpus_mod.db_word_spans(ctx.wakeword.name)
    segments = corpus_mod.list_segments(ctx.wakeword.name, require_vtt=False)

    # Occurrences déjà retenues comme positifs (t_start encodé dans le nom)
    known: dict[str, list[float]] = {}
    if only_known:
        import re
        splits = ctx.word_splits()
        for split_files in splits.values():
            for f in split_files.get(ctx.wakeword.name, []):
                m = re.match(rf"yt_({corpus_mod.VIDEO_ID})_.*_(\d+(?:\.\d+)?)s", f.stem)
                if m:
                    known.setdefault(m.group(1), []).append(float(m.group(2)))

    for s in ("train", "val", "test"):
        (out_root / s).mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    skipped: Counter = Counter()
    for seg in segments:
        split = vid_split.get(seg.video_id)
        if split is None:                    # vidéo inconnue → réservée au banc
            continue
        seg_spans = [sp for sp in spans.get(seg.video_id, [])
                     if seg.start <= sp[0] and sp[1] <= seg.end]
        if not seg_spans:
            continue
        try:
            audio, sr = sf.read(seg.wav, dtype="float32")
        except Exception:
            skipped["illisible"] += 1
            continue
        if sr != ctx.sr:
            skipped["sample_rate"] += 1
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        for t0, t1, _surface in seg_spans:
            if only_known and not any(abs(t0 - k) <= 0.35
                                      for k in known.get(seg.video_id, [])):
                skipped["hors_splits"] += 1
                continue
            if t1 - t0 > clip_s:
                skipped["mot_trop_long"] += 1
                continue
            # Marge déterministe par occurrence (reproductible à seed donnée)
            rng = random.Random(f"{ctx.seed}|{seg.video_id}|{round(t0 * 100)}")
            margin = rng.uniform(0.0, jitter_s)
            end_local = (t1 - seg.start) + margin
            start_local = end_local - clip_s
            if start_local < 0:              # pas assez de contexte avant :
                start_local, end_local = 0.0, clip_s      # on recale au début…
                if (t1 - seg.start) > end_local:          # …sauf si le mot déborde
                    skipped["sans_contexte"] += 1
                    continue
            i0 = int(start_local * ctx.sr)
            win = audio[i0:i0 + n]
            if len(win) < n:                 # fin de segment trop proche
                skipped["fin_de_segment"] += 1
                continue
            name = f"yt_{seg.video_id}_recut_{t0:.2f}s.wav"
            sf.write(out_root / split / name, win, ctx.sr, subtype="PCM_16")
            counts[split] += 1

    print(f"    {src.name} : " + "  ".join(f"{s}:{counts[s]}" for s in ("train", "val", "test"))
          + (f"  écartés {dict(skipped)}" if skipped else ""))
    if not sum(counts.values()):
        raise RuntimeError(f"{src.name} : aucune fenêtre produite — corpus ou "
                           "discovery.db absents ?")
    return mark_done(out_root)
