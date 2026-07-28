"""Front-end acoustique — **implémentation unique** du log-mel.

Dans le projet précédent, `get_spectrogram()` était recopié dans 6 scripts
(entraînement, live, guidé, sweep, banc, erreurs). Toute divergence entre la
version d'entraînement et celle d'inférence produit un modèle qui « marche en
test et rate en vrai », et c'est indétectable par les métriques. Ici il n'y a
qu'une seule fonction, paramétrée par `AudioConfig`, utilisée partout.

Chaîne : waveform 1 s → STFT → magnitude → banc de 40 filtres mel → log →
z-score PAR EXEMPLE (robuste au niveau d'enregistrement) → (T, mel, 1).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..config import AudioConfig, WakewordConfig


def mel_matrix(cfg: AudioConfig, sample_rate: int, num_spectrogram_bins: int) -> tf.Tensor:
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=cfg.num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=cfg.lower_edge_hertz,
        upper_edge_hertz=cfg.upper_edge_hertz or sample_rate / 2,
    )


def log_mel(waveform, cfg: AudioConfig, sample_rate: int):
    """Log-mel normalisé. Accepte une waveform (T,) ou un lot (B, T)."""
    waveform = tf.cast(waveform, tf.float32)
    stft = tf.signal.stft(waveform, frame_length=cfg.frame_length, frame_step=cfg.frame_step)
    spec = tf.abs(stft)
    mel = mel_matrix(cfg, sample_rate, spec.shape[-1])
    mel_spec = tf.tensordot(spec, mel, axes=1)
    mel_spec.set_shape(spec.shape[:-1].concatenate(mel.shape[-1:]))
    logm = tf.math.log(mel_spec + 1e-6)
    # z-score par exemple : le modèle ne doit pas apprendre le gain du micro
    mean = tf.reduce_mean(logm, axis=[-2, -1], keepdims=True)
    std = tf.math.reduce_std(logm, axis=[-2, -1], keepdims=True)
    return ((logm - mean) / (std + 1e-6))[..., tf.newaxis]


class FeatureExtractor:
    """Enveloppe pratique liée à un mot-clé (sert au train, au live et au banc)."""

    def __init__(self, wakeword: WakewordConfig):
        self.wakeword = wakeword
        self.cfg = wakeword.audio
        self.sr = wakeword.sample_rate

    def __call__(self, waveform):
        return log_mel(waveform, self.cfg, self.sr)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Forme d'entrée du modèle, déduite analytiquement (pas d'appel TF)."""
        n = self.wakeword.clip_samples
        frames = 1 + (n - self.cfg.frame_length) // self.cfg.frame_step
        return (frames, self.cfg.num_mel_bins, 1)

    def batch(self, windows: np.ndarray) -> tf.Tensor:
        return log_mel(tf.constant(windows, tf.float32), self.cfg, self.sr)


def sliding_windows(audio: np.ndarray, clip_samples: int, hop_samples: int):
    """Découpe un signal continu en fenêtres glissantes.

    Renvoie (windows (N, clip_samples), peaks (N,), starts (N,) en échantillons).
    Le pic d'amplitude sert de portail d'énergie : identique en live et au banc.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < clip_samples:
        audio = np.pad(audio, (0, clip_samples - len(audio)))
    starts = np.arange(0, len(audio) - clip_samples + 1, hop_samples)
    windows = np.stack([audio[s:s + clip_samples] for s in starts])
    return windows, np.abs(windows).max(axis=1), starts
