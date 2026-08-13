"""Banc d'essai STREAMING — le juge de paix du projet.

Pourquoi il existe : le test par clips a **mal classé** les modèles. Le meilleur
en F1 par clip s'est révélé le pire en conditions réelles ; un modèle à F1 0.95
ratait une occurrence sur trois dans un vrai flux. Un clip de 1 s parfaitement
centré n'existe pas en production — en production il y a des demi-mots, du
recouvrement, du bruit, et une décision à prendre 8 fois par seconde.

Protocole : on rejoue la logique live exacte sur des segments YouTube continus
jamais vus à l'entraînement, avec pour vérité terrain les alignements WhisperX.
Mesures : **rappel streaming** (occurrences réellement attrapées) et
**fausses alarmes par heure** — les deux seuls chiffres qui décrivent
l'expérience d'usage d'un détecteur always-on.

Interprétation des FA/h : le corpus est thématique (des vidéos où l'on PARLE
d'éloquence), c'est donc un pire cas volontaire, pas une moyenne de la vie
courante. Les événements « incertains » sont exclus du décompte.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import soundfile as sf

from .. import runtime
from ..config import WakewordConfig
from ..data import corpus as corpus_mod
from ..inference.detector import load_detector

MATCH_BEFORE_S = 0.5      # un trigger peut précéder légèrement le début du mot
MATCH_AFTER_S = 2.0       # …ou suivre, le temps que la fenêtre couvre le mot entier


def eligible_segments(wakeword: str, forbidden_videos: set[str], minutes: float,
                      seed: int = 42, uncertain_s: float = 5.0) -> list:
    """Sélection reproductible : ~moitié de segments contenant le mot, moitié
    sans, une seule occurrence par vidéo, et **jamais** une vidéo vue à
    l'entraînement (sinon on mesurerait de la mémorisation).

    Les segments ADDITIONNELS (`data/external/bench_extra/`, extension ROADMAP
    P0 — SUMM-RE etc.) passent en premier : ils ne viennent jamais de vidéos
    d'entraînement et leur vérité terrain est vérifiée. Le budget `minutes`
    restant est rempli par le corpus YouTube comme avant."""
    import random

    extra, used = [], 0.0
    for s in corpus_mod.list_extra_segments(wakeword):
        if used + s.duration > minutes * 60:     # le budget --minutes est un plafond
            break
        extra.append(s)
        used += s.duration
    budget_left = max(0.0, minutes * 60 - used)

    segments = [s for s in corpus_mod.list_segments(wakeword, uncertain_s=uncertain_s)
                if s.video_id not in forbidden_videos]
    rng = random.Random(seed)
    with_occ = [s for s in segments if s.occurrences]
    without = [s for s in segments if not s.occurrences]
    rng.shuffle(with_occ)
    rng.shuffle(without)
    # Le mot NU (« éloquence » sans élision) est rarissime dans le corpus
    # éligible (la plupart de ses vidéos sont parties au train) : les segments
    # qui en contiennent passent EN TÊTE de la sélection, sinon le banc ne
    # mesure que les formes élidées (constat du 2026-08-13 : 0 nu sur 27).
    with_occ.sort(key=lambda s: not any(
        corpus_mod.surface_form(x, wakeword) == "nu" for x in (s.surfaces or [])))

    picked, seen = list(extra), set()
    for pool in (with_occ, without):
        budget, used = budget_left / 2, 0.0
        for s in pool:
            if s.video_id in seen or used >= budget:
                continue
            picked.append(s)
            seen.add(s.video_id)
            used += s.duration
    return picked


def forbidden_from_splits(splits_csv: Path, keep: tuple[str, ...] = ("train", "val")) -> set[str]:
    """Vidéos interdites au banc = celles qui ont servi à entraîner."""
    import csv

    out = set()
    if not splits_csv.exists():
        return out
    with open(splits_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("source") == "youtube" and row["split"] in keep:
                m = re.match(rf"yt_({corpus_mod.VIDEO_ID})", row["group"])
                if m:
                    out.add(m.group(1))
    return out


def score(triggers: list[float], occurrences: list[float], uncertain: list[float],
          uncertain_s: float = 5.0) -> dict:
    """Apparie déclenchements ↔ occurrences. Un trigger non apparié proche d'un
    temps VTT est classé « incertain » : ni détection, ni fausse alarme."""
    hit = [False] * len(occurrences)
    false_alarms, unknown, events = 0, 0, []
    for t in triggers:
        matched = [j for j, o in enumerate(occurrences)
                   if o - MATCH_BEFORE_S <= t <= o + MATCH_AFTER_S]
        for j in matched:
            hit[j] = True
        if matched:
            events.append({"t": round(t, 2), "kind": "TP"})
        elif any(abs(t - u) <= uncertain_s for u in uncertain):
            unknown += 1
            events.append({"t": round(t, 2), "kind": "INCERTAIN"})
        else:
            false_alarms += 1
            events.append({"t": round(t, 2), "kind": "FA"})
    events += [{"t": round(o, 2), "kind": "FN"} for o, h in zip(occurrences, hit) if not h]
    return {"n_occ": len(occurrences), "detected": sum(hit), "false_alarms": false_alarms,
            "uncertain": unknown, "hits": hit,
            "events": sorted(events, key=lambda e: e["t"])}


def run(models: dict[str, Path], wakeword: WakewordConfig, minutes: float = 16.0,
        thresholds: tuple[float, ...] = (0.5, 0.8), splits_csv: Path | None = None,
        seed: int = 42, collect_events: bool = True, use_gpu: bool = False) -> dict:
    # CPU par défaut : sur Metal, les probabilités divergent assez pour changer
    # les détections (mesuré : 4/5 occurrences attrapées sur CPU, 0/5 sur GPU
    # avec le même modèle et le même audio). Voir ADR-002.
    runtime.configure(use_gpu=use_gpu)
    forbidden = forbidden_from_splits(splits_csv) if splits_csv else set()
    # Vidéos sacrifiées aux hard negatives : entrées à l'entraînement via la
    # source `hard_negatives`, donc interdites au banc pour toujours.
    import json as _json

    from .. import paths as _paths
    reg = _paths.word_dir(wakeword.name) / "hard_negative_videos.json"
    if reg.exists():
        sacrificed = set(_json.loads(reg.read_text()))
        if sacrificed:
            print(f"🚫  {len(sacrificed)} vidéo(s) sacrifiées aux hard negatives — "
                  "exclues du banc")
        forbidden |= sacrificed
    segments = eligible_segments(wakeword.name, forbidden, minutes, seed)
    if not segments:
        raise RuntimeError("aucun segment éligible — corpus absent ou tout interdit ?")

    total_s = sum(s.duration for s in segments)
    n_occ = sum(len(s.occurrences) for s in segments)
    print(f"🎬  {len(segments)} segments · {total_s / 60:.1f} min · {n_occ} occurrences "
          f"(vérité WhisperX) · {len(forbidden)} vidéos interdites (vues à l'entraînement)")

    results: dict = {}
    for name, path in models.items():
        print(f"\n📦  {name} — {path}")
        if str(path).endswith(".onnx"):        # concurrent openWakeWord (P3)
            from .oww_adapter import OwwDetector
            detector = OwwDetector(Path(path), wakeword)
        else:
            detector = load_detector(path, wakeword)
        agg = {th: {"n_occ": 0, "detected": 0, "false_alarms": 0, "uncertain": 0}
               for th in thresholds}
        # Rappel PAR FORME (nu / l' / d') : 85 % des occurrences réelles sont
        # élidées — il faut voir si le modèle est plus faible sur une forme.
        forms: dict = {th: {} for th in thresholds}
        events: dict = {th: [] for th in thresholds}
        for seg in segments:
            audio, sr = sf.read(seg.wav, dtype="float32")
            if sr != wakeword.sample_rate:
                print(f"    ⚠️  {seg.wav.name} : {sr} Hz ≠ {wakeword.sample_rate} — ignoré")
                continue
            probas, peaks, _ = detector.window_probas(audio)
            seg_forms = [corpus_mod.surface_form(s, wakeword.name)
                         for s in (seg.surfaces or [wakeword.name] * len(seg.occurrences))]
            for th in thresholds:
                sc = score(detector.triggers_from(probas, peaks, th),
                           seg.occurrences, seg.uncertain)
                for k in agg[th]:
                    agg[th][k] += sc[k]
                for form, h in zip(seg_forms, sc["hits"]):
                    n, d = forms[th].get(form, (0, 0))
                    forms[th][form] = (n + 1, d + int(h))
                if collect_events:
                    events[th] += [{**e, "segment": seg.wav.name, "video": seg.video_id}
                                   for e in sc["events"] if e["kind"] != "TP"]

        results[name] = {}
        for th in thresholds:
            a = agg[th]
            recall = a["detected"] / a["n_occ"] if a["n_occ"] else float("nan")
            fa_h = a["false_alarms"] / (total_s / 3600)
            by_form = {form: {"n_occ": n, "detected": d,
                              "recall": d / n if n else float("nan")}
                       for form, (n, d) in sorted(forms[th].items())}
            results[name][f"th{th}"] = {
                "recall_stream": recall, "frr_stream": 1 - recall,
                "false_alarms": a["false_alarms"], "fa_per_hour": fa_h,
                "uncertain": a["uncertain"], "n_occ": a["n_occ"], "detected": a["detected"],
                "recall_by_form": by_form,
                "events": events[th] if collect_events else [],
            }
            detail = " · ".join(f"{form} {v['detected']}/{v['n_occ']}"
                                for form, v in by_form.items())
            print(f"    seuil {th} : rappel {recall:6.1%} ({a['detected']}/{a['n_occ']}) · "
                  f"FA {a['false_alarms']} → {fa_h:.1f}/h · incertains {a['uncertain']}"
                  f"{'  [' + detail + ']' if detail else ''}")

    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "wakeword": wakeword.name,
        "total_seconds": total_s,
        "n_occurrences": n_occ,
        "live_logic": wakeword.live.model_dump(),
        "match_window_s": [MATCH_BEFORE_S, MATCH_AFTER_S],
        "ground_truth": "discovery.db (WhisperX) ; VTT = zones incertaines",
        "forbidden_videos": sorted(forbidden),
        "segments": [{k: (str(v) if isinstance(v, Path) else v)
                      for k, v in asdict(s).items() if k != "vtt"} for s in segments],
        "models": {k: str(v) for k, v in models.items()},
        "results": results,
    }
