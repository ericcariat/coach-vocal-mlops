"""API FastAPI — le modèle exposé comme un service (Swagger sur `/docs`).

Sépare nettement les deux questions qu'on pose à un détecteur :
- `/predict` : « ce clip contient-il le mot ? » (probabilité brute) ;
- `/detect` : « où le détecteur se serait-il réveillé dans ce flux ? »
  (machine à états live complète, avec seuil, fenêtres consécutives et cooldown).

Le modèle est chargé **paresseusement** et mis en cache par (mot-clé, run) : le
démarrage reste instantané et le premier appel paie le warm-up, une seule fois.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from .. import paths, registry, runtime
from ..config import list_experiments, load_wakeword

api = FastAPI(
    title="Coach vocal — API wake word",
    version="0.1.0",
    description=(
        "Détection de mot-clé always-on. `/predict` renvoie une probabilité, "
        "`/detect` rejoue la logique temps réel complète. Le modèle servi est le "
        "**champion** du registre, sauf `run` explicite."),
)

_CACHE: dict[tuple[str, Optional[str]], object] = {}


def get_detector(wakeword: str, run: str | None = None):
    key = (wakeword, run)
    if key not in _CACHE:
        from ..inference.detector import load_detector

        runtime.configure(use_gpu=False)         # Metal fausse les probas (ADR-002)
        try:
            word = load_wakeword(wakeword)
            model_path = registry.model_path(wakeword, run)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        _CACHE[key] = load_detector(model_path, word)
    return _CACHE[key]


def read_audio(raw: bytes, target_sr: int) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        raise HTTPException(400, f"fichier audio illisible : {exc}") from exc
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ── Schémas de réponse ────────────────────────────────────────────────────────
class Health(BaseModel):
    status: str
    wakewords: list[str]
    experiments: list[str]


class Prediction(BaseModel):
    wakeword: str
    run: str
    probability: float
    detected: bool
    threshold: float
    peak: float
    n_windows: int
    inference_ms: float


class Trigger(BaseModel):
    t: float
    label: str = "trigger"


class Detection(BaseModel):
    wakeword: str
    run: str
    threshold: float
    duration_s: float
    triggers: list[Trigger]
    live_logic: dict


# ── Points d'entrée ───────────────────────────────────────────────────────────
@api.get("/health", response_model=Health, tags=["système"])
def health():
    words = sorted(p.stem for p in (paths.CONFIGS / "wakeword").glob("*.yaml"))
    return Health(status="ok", wakewords=words, experiments=list_experiments())


@api.get("/models", tags=["registre"])
def models(wakeword: str = "eloquence"):
    """Tous les runs, leurs métriques, et le champion courant avec sa justification."""
    reg = registry.load(wakeword)
    return {"champion": reg.get("champion"), "history": reg.get("history", []),
            "runs": registry.list_runs(wakeword)}


@api.get("/metrics", tags=["registre"])
def metrics(wakeword: str = "eloquence"):
    """Métriques par clip + banc streaming — ce qu'affiche le dashboard."""
    from ..evaluation.dashboard import export_json

    return export_json(wakeword)


@api.get("/config", tags=["système"])
def config(wakeword: str = "eloquence"):
    """Configuration servie (front-end acoustique et logique live)."""
    return load_wakeword(wakeword).model_dump()


@api.post("/predict", response_model=Prediction, tags=["inférence"])
def predict(file: UploadFile = File(..., description="WAV mono (toute durée ≥ 1 s)"),
            wakeword: str = Query("eloquence"), run: Optional[str] = None,
            threshold: Optional[float] = None):
    """Probabilité maximale sur les fenêtres glissantes du fichier."""
    detector = get_detector(wakeword, run)
    audio = read_audio(file.file.read(), detector.wakeword.sample_rate)
    th = threshold if threshold is not None else detector.threshold

    t0 = time.perf_counter()
    probas, peaks, _ = detector.window_probas(audio)
    elapsed = (time.perf_counter() - t0) * 1000
    i = int(np.argmax(probas))
    return Prediction(
        wakeword=wakeword, run=run or registry.champion_run(wakeword) or "?",
        probability=float(probas[i]), detected=bool(probas[i] > th), threshold=th,
        peak=float(peaks[i]), n_windows=len(probas), inference_ms=round(elapsed, 1))


@api.post("/detect", response_model=Detection, tags=["inférence"])
def detect(file: UploadFile = File(..., description="WAV mono, flux continu"),
           wakeword: str = Query("eloquence"), run: Optional[str] = None,
           threshold: Optional[float] = None):
    """Rejoue la logique live : renvoie les instants de réveil du détecteur."""
    detector = get_detector(wakeword, run)
    audio = read_audio(file.file.read(), detector.wakeword.sample_rate)
    th = threshold if threshold is not None else detector.threshold
    triggers = detector.run_offline(audio, th)
    return Detection(
        wakeword=wakeword, run=run or registry.champion_run(wakeword) or "?",
        threshold=th, duration_s=round(len(audio) / detector.wakeword.sample_rate, 2),
        triggers=[Trigger(t=round(t, 3)) for t in triggers],
        live_logic=detector.live.model_dump())
