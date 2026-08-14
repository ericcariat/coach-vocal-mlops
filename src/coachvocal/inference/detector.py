"""Détecteur always-on — **une seule** machine à états, partagée live/banc.

C'est le point le plus important de l'architecture d'inférence : le banc doit
mesurer EXACTEMENT ce que fait le micro, sinon il mesure autre chose. Les deux
chemins (temps réel `push()` et hors-ligne `run_offline()`) appliquent donc la
même règle de décision, dans le même code.

Règle de décision (chaque fenêtre de 1 s, toutes les 125 ms) :
1. **Portail d'énergie** : pic < `min_peak` → fenêtre ignorée. Empêche le
   modèle de halluciner sur du bruit de plancher amplifié par le z-score.
2. **N fenêtres consécutives** au-dessus du seuil. Un pic isolé de proba est du
   bruit ; un vrai mot reste visible plusieurs fenêtres d'affilée. C'est le
   réglage qui a le plus fait baisser les fausses alarmes.
3. **Cooldown** après déclenchement : un mot prononcé une fois ne doit compter
   qu'une fois.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from ..audio.features import FeatureExtractor, sliding_windows
from ..config import WakewordConfig


class WakeWordDetector:
    def __init__(self, model, wakeword: WakewordConfig, threshold: float | None = None):
        self.model = model
        self.wakeword = wakeword
        self.live = wakeword.live
        self.threshold = wakeword.live.threshold if threshold is None else threshold
        self.features = FeatureExtractor(wakeword)
        self.n = wakeword.clip_samples
        self.hop = int(self.live.hop_s * wakeword.sample_rate)
        self._buffer = deque(maxlen=self.n)
        self._buffer.extend(np.zeros(self.n, np.float32))
        self._consecutive = 0
        self._cooldown_until = -1.0
        self._t = 0.0

    # ── Chemin hors-ligne (banc, analyse de fichiers) ──────────────────────────
    def window_probas(self, audio: np.ndarray, batch_size: int = 64):
        windows, peaks, starts = sliding_windows(audio, self.n, self.hop)
        specs = self.features.batch(windows)
        probas = self.model.predict(specs, verbose=0, batch_size=batch_size).ravel()
        return probas, peaks, starts

    def triggers_from(self, probas, peaks, threshold: float | None = None) -> list[float]:
        """Rejoue la machine à états sur une séquence complète de fenêtres.
        Renvoie les instants (s, fin de fenêtre) des déclenchements."""
        th = self.threshold if threshold is None else threshold
        trig: list[float] = []
        consecutive = 0
        cooldown_until = -1.0
        clip_s = self.wakeword.audio.clip_seconds
        for i, (p, peak) in enumerate(zip(probas, peaks)):
            t = i * self.live.hop_s + clip_s
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

    def run_offline(self, audio: np.ndarray, threshold: float | None = None) -> list[float]:
        probas, peaks, _ = self.window_probas(audio)
        return self.triggers_from(probas, peaks, threshold)

    # ── Chemin temps réel (micro) ─────────────────────────────────────────────
    def push(self, chunk: np.ndarray) -> dict | None:
        """Consomme un bloc de `hop` échantillons ; renvoie l'événement de
        déclenchement le cas échéant. Même règle que `triggers_from`."""
        self._buffer.extend(chunk.astype(np.float32).ravel())
        self._t += len(chunk) / self.wakeword.sample_rate
        window = np.array(self._buffer, dtype=np.float32)
        peak = float(np.abs(window).max())
        proba = float(self.model(self.features(window)[np.newaxis, ...], training=False)[0, 0])

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

    def reset(self) -> None:
        self._buffer.clear()
        self._buffer.extend(np.zeros(self.n, np.float32))
        self._consecutive = 0
        self._cooldown_until = -1.0
        self._t = 0.0


def load_detector(model_path: Path, wakeword: WakewordConfig,
                  threshold: float | None = None):
    """Charge le détecteur du champion, quel que soit son front-end.

    Depuis la promotion de la tête openWakeWord (ADR-008), un champion peut
    être un `.keras` (notre CNN, front-end log-mel maison) ou un `.onnx`
    (tête sur extracteur Google gelé, front-end openWakeWord). Les deux
    exposent la même interface : `window_probas`, `triggers_from`, `push`.
    """
    if str(model_path).endswith(".onnx"):
        from ..evaluation.oww_adapter import OwwDetector
        return OwwDetector(Path(model_path), wakeword, threshold)

    import tensorflow as tf

    from ..models import activations  # noqa: F401 — enregistre relu_max avant load_model (ADR-002)

    model = tf.keras.models.load_model(model_path)
    detector = WakeWordDetector(model, wakeword, threshold)
    detector.model(detector.features(np.zeros(detector.n, np.float32))[np.newaxis, ...],
                   training=False)          # warm-up : évite un pic de latence au 1er mot
    return detector
