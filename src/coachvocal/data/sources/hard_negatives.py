"""Hard negatives : les fausses alarmes RÉELLES, confirmées à l'oreille.

La boucle complète du projet : le banc détecte → l'humain juge (page « Banc
streaming », verdicts persistés) → les FA confirmées deviennent des négatifs
d'entraînement. Ce sont les négatifs les plus précieux qui existent : le modèle
s'est réellement trompé dessus, en conditions réelles.

Anti-fuite, en deux temps :
- seules les FA des segments YouTube sont extraites (les FA SUMM-RE
  imposeraient de sacrifier les réunions du banc — leur domaine est couvert
  par `summre_train`) ;
- les vidéos utilisées sont inscrites dans `hard_negative_videos.json`, que le
  banc ajoute à sa liste d'interdits : une vidéo sacrifiée à l'entraînement ne
  re-servira JAMAIS à mesurer.

Extraction : la fenêtre exacte qui a déclenché ([t−1 s, t]) + deux décalages
(±0,25 s) pour couvrir le voisinage — 3 clips par FA confirmée.
"""

from __future__ import annotations

import json
from collections import Counter

import soundfile as sf

from ... import paths
from ...config import SourceConfig
from .. import corpus as corpus_mod
from . import SourceContext, SplitPools, cached, mark_done, source

VIDEOS_FILE = "hard_negative_videos.json"


@source("hard_negatives")
def hard_negatives(src: SourceConfig, ctx: SourceContext) -> SplitPools:
    """Params : `model` (dont on lit les verdicts, défaut v11_speech_300),
    `threshold` (défaut th0.5), `offsets_s` (défaut [-0.25, 0, 0.25])."""
    out_root = ctx.cache(src.name)
    if (hit := cached(out_root)) is not None:
        return hit

    model = src.params.get("model", "v11_speech_300")
    th = src.params.get("threshold", "th0.5")
    offsets = [float(o) for o in src.params.get("offsets_s", [-0.25, 0.0, 0.25])]

    verdicts_path = paths.report_dir("stream_bench") / f"{ctx.wakeword.name}_verdicts.json"
    if not verdicts_path.exists():
        raise RuntimeError("hard_negatives : aucun verdict — juger les FA dans "
                           "la page « Banc streaming » d'abord")
    verdicts = json.loads(verdicts_path.read_text())

    confirmed = []                       # [(segment_wav_name, t_local)]
    for key, v in verdicts.items():
        vmodel, vth, seg, t, kind = key.split("|")
        if (vmodel == model and vth == th and kind == "FA"
                and v.get("verdict", "").startswith("✅")
                and not seg.startswith("summre")):
            confirmed.append((seg, float(t)))
    if not confirmed:
        raise RuntimeError(f"hard_negatives : aucune FA confirmée pour {model}/{th}")

    (out_root / "train").mkdir(parents=True, exist_ok=True)
    n = ctx.clip_samples
    counts: Counter = Counter()
    videos = set()
    for seg_name, t in sorted(confirmed):
        wav = corpus_mod.CORPUS / "audio" / seg_name
        if not wav.exists():
            counts["introuvable"] += 1
            continue
        audio, sr = sf.read(wav, dtype="float32")
        if sr != ctx.sr:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        videos.add(seg_name.rsplit("_", 1)[0])
        for off in offsets:
            end = int((t + off) * ctx.sr)
            start = end - n
            if start < 0 or end > len(audio):
                continue
            name = f"hn_{seg_name[:-4]}_{t:.2f}s_{off:+.2f}.wav"
            sf.write(out_root / "train" / name, audio[start:end], ctx.sr,
                     subtype="PCM_16")
            counts["train"] += 1

    # Registre des vidéos sacrifiées : le banc les interdit désormais.
    reg_path = paths.word_dir(ctx.wakeword.name) / VIDEOS_FILE
    known = set(json.loads(reg_path.read_text())) if reg_path.exists() else set()
    reg_path.write_text(json.dumps(sorted(known | videos), indent=1))

    print(f"    {src.name} : train:{counts['train']} "
          f"({len(confirmed)} FA confirmées, {len(videos)} vidéos sacrifiées"
          + (f", {counts['introuvable']} segments introuvables" if counts["introuvable"] else "")
          + ")")
    if not counts["train"]:
        raise RuntimeError(f"{src.name} : aucune fenêtre extraite")
    return mark_done(out_root)
