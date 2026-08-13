"""Adaptateur openWakeWord → banc streaming (ROADMAP P3).

Fait tourner un modèle openWakeWord (tête ONNX entraînée ailleurs) sur NOTRE
banc, avec NOTRE règle de décision — la seule comparaison honnête (cf. les
quatre études : « mêmes flux, même machine à états, mêmes règles de comptage »).

Chaîne openWakeWord (front-end distinct du nôtre, et c'est voulu — chaque
système garde le sien) :
  audio 16 kHz → melspectrogram.onnx (32 bandes, 1 trame/10 ms, mise à
  l'échelle x/10+2) → embedding_model.onnx (fenêtre 76 trames, pas 8 → un
  vecteur de 96 toutes les 80 ms) → tête [1,16,96] → probabilité.
Cadence des scores : 80 ms (contre 125 ms chez nous). La règle des N fenêtres
consécutives est donc plus courte en TEMPS à N égal — comparaison à lire avec
cette réserve, documentée dans le rapport du banc.

Front-ends : `data/external/oww_models/` (release v0.5.1 du dépôt officiel).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths
from ..config import WakewordConfig

OWW_DIR = paths.EXTERNAL / "oww_models"
MEL_HOP_S = 0.010          # une trame mel / 10 ms
EMB_STEP_FRAMES = 8        # un embedding / 8 trames = 80 ms
EMB_WIN_FRAMES = 76
HEAD_EMBEDDINGS = 16       # fenêtre de la tête ≈ 76+15*8 trames ≈ 1.96 s


class OwwDetector:
    """Même interface que WakeWordDetector côté banc : `window_probas` +
    `triggers_from` (portail d'énergie, N consécutives, cooldown)."""

    def __init__(self, head_path: Path, wakeword: WakewordConfig,
                 threshold: float | None = None):
        import onnxruntime as ort

        opts = {"providers": ["CPUExecutionProvider"]}
        self._mel = ort.InferenceSession(str(OWW_DIR / "melspectrogram.onnx"), **opts)
        self._emb = ort.InferenceSession(str(OWW_DIR / "embedding_model.onnx"), **opts)
        self._head = ort.InferenceSession(str(head_path), **opts)
        self._head_in = self._head.get_inputs()[0].name
        self.wakeword = wakeword
        self.live = wakeword.live
        self.threshold = wakeword.live.threshold if threshold is None else threshold
        self.hop_s = MEL_HOP_S * EMB_STEP_FRAMES              # 80 ms
        self.window_s = MEL_HOP_S * (EMB_WIN_FRAMES + (HEAD_EMBEDDINGS - 1) * EMB_STEP_FRAMES)

    def window_probas(self, audio: np.ndarray, batch_size: int = 64):
        sr = self.wakeword.sample_rate
        mel = self._mel.run(None, {"input": audio[np.newaxis, :].astype(np.float32)})[0]
        frames = np.squeeze(mel) / 10.0 + 2.0                 # mise à l'échelle oWW
        if frames.ndim != 2 or len(frames) < EMB_WIN_FRAMES:
            return np.zeros(0), np.zeros(0), np.zeros(0, int)

        # Embeddings : fenêtres de 76 trames, pas de 8
        starts_f = range(0, len(frames) - EMB_WIN_FRAMES + 1, EMB_STEP_FRAMES)
        windows = np.stack([frames[i:i + EMB_WIN_FRAMES] for i in starts_f])
        embs = []
        for i in range(0, len(windows), batch_size):
            out = self._emb.run(None, {"input_1": windows[i:i + batch_size,
                                                          :, :, np.newaxis].astype(np.float32)})[0]
            embs.append(out.reshape(len(out), 96))
        embs = np.concatenate(embs)
        if len(embs) < HEAD_EMBEDDINGS:
            return np.zeros(0), np.zeros(0), np.zeros(0, int)

        # Tête : fenêtre glissante de 16 embeddings, pas de 1 (→ 1 score / 80 ms)
        probas = np.empty(len(embs) - HEAD_EMBEDDINGS + 1, np.float32)
        for j in range(len(probas)):
            seq = embs[j:j + HEAD_EMBEDDINGS][np.newaxis, ...].astype(np.float32)
            probas[j] = float(self._head.run(None, {self._head_in: seq})[0].ravel()[0])

        # Pic d'énergie de la fenêtre audio correspondante (portail du live)
        win = int(self.window_s * sr)
        hop = int(self.hop_s * sr)
        starts = np.arange(len(probas)) * hop
        peaks = np.array([float(np.abs(audio[s:s + win]).max() or 0.0) for s in starts])
        return probas, peaks, starts

    # ── Chemin temps réel (page Démo) — recalcul complet par pas de 80 ms,
    # comme le runtime officiel (stateless). Même règle de décision que push()
    # du détecteur maison.
    @property
    def hop(self) -> int:
        return int(self.hop_s * self.wakeword.sample_rate)

    def push(self, chunk: np.ndarray) -> dict | None:
        from collections import deque

        sr = self.wakeword.sample_rate
        # Tampon > fenêtre de la tête : il faut 196 trames mel PLEINES (16
        # embeddings), soit un peu plus d'audio que window_s — 1,96 s donnait
        # ~194 trames et la tête ne tournait jamais (proba 0 permanente).
        n = int((self.window_s + 0.1) * sr)
        if not hasattr(self, "_buffer"):
            self._buffer = deque(maxlen=n)
            self._buffer.extend(np.zeros(n, np.float32))
            self._consecutive = 0
            self._cooldown_until = -1.0
            self._t = 0.0
        self._buffer.extend(chunk.astype(np.float32).ravel())
        self._t += len(chunk) / sr
        window = np.array(self._buffer, dtype=np.float32)
        peak = float(np.abs(window).max())

        mel = self._mel.run(None, {"input": window[np.newaxis, :]})[0]
        frames = (np.squeeze(mel) / 10.0 + 2.0)[-((HEAD_EMBEDDINGS - 1) * EMB_STEP_FRAMES
                                                  + EMB_WIN_FRAMES):]
        wins = np.stack([frames[i:i + EMB_WIN_FRAMES]
                         for i in range(0, len(frames) - EMB_WIN_FRAMES + 1, EMB_STEP_FRAMES)])
        embs = self._emb.run(None, {"input_1": wins[:, :, :, np.newaxis].astype(np.float32)})[0]
        embs = embs.reshape(len(embs), 96)[-HEAD_EMBEDDINGS:]
        if len(embs) < HEAD_EMBEDDINGS:
            return {"proba": 0.0, "peak": peak, "triggered": False, "t": self._t}
        proba = float(self._head.run(None, {self._head_in:
                      embs[np.newaxis, ...].astype(np.float32)})[0].ravel()[0])

        if self._t < self._cooldown_until:
            self._consecutive = 0
            return {"proba": proba, "peak": peak, "triggered": False, "t": self._t}
        fired = (peak >= self.live.min_peak) and (proba > self.threshold)
        self._consecutive = self._consecutive + 1 if fired else 0
        triggered = self._consecutive >= self.live.n_consecutive
        if triggered:
            self._consecutive = 0
            self._cooldown_until = self._t + self.live.cooldown_s
        return {"proba": proba, "peak": peak, "triggered": triggered, "t": self._t}

    def triggers_from(self, probas, peaks, threshold: float | None = None) -> list[float]:
        """La MÊME règle que WakeWordDetector, à la cadence oWW (80 ms)."""
        th = self.threshold if threshold is None else threshold
        trig: list[float] = []
        consecutive = 0
        cooldown_until = -1.0
        for i, (p, peak) in enumerate(zip(probas, peaks)):
            t = i * self.hop_s + self.window_s
            if t < cooldown_until:
                consecutive = 0
                continue
            fired = (peak >= self.live.min_peak) and (p > th)
            consecutive = consecutive + 1 if fired else 0
            if consecutive >= self.live.n_consecutive:
                trig.append(t)
                consecutive = 0
                cooldown_until = t + self.live.cooldown_s
        return trig


def frontends_available() -> bool:
    return (OWW_DIR / "melspectrogram.onnx").exists() and \
           (OWW_DIR / "embedding_model.onnx").exists()
