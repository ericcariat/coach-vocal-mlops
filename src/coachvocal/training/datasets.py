"""Manifest → `tf.data.Dataset`.

Chaîne : chemins → WAV décodé (padding/troncature à la longueur du clip) →
augmentation (train uniquement) → log-mel → batch.

L'augmentation de VITESSE (×0.85–1.15) vient du diagnostic du run v01 : 15
« éloquence » sur 20 prononcés au tempo réel étaient ratés, mais 20/20 reconnus
une fois ralentis de 8-15 %. Le modèle avait appris un débit, pas un mot.
Le décalage temporel est NON circulaire (on ne recolle pas la fin au début, ce
qui fabriquerait des mots impossibles).
"""

from __future__ import annotations

import tensorflow as tf

from ..audio.features import FeatureExtractor
from ..config import AugmentationConfig, WakewordConfig


def _decode(path: tf.Tensor, label: tf.Tensor, clip_samples: int):
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1,
                                   desired_samples=clip_samples)
    return tf.squeeze(audio, axis=-1), label


def _speed(audio, labels, clip_samples: int, lo: float, hi: float):
    """Rééchantillonnage plus proche voisin, mot calé au début (étirement depuis
    l'origine) : rapide, sans dépendance, suffisant à ±15 %."""
    batch = tf.shape(audio)[0]
    factors = tf.random.uniform([batch], lo, hi)
    max_src = int(clip_samples / lo) + 1
    padded = tf.pad(audio, [[0, 0], [0, max_src - clip_samples]])
    idx = tf.cast(tf.cast(tf.range(clip_samples)[tf.newaxis, :], tf.float32)
                  / factors[:, tf.newaxis], tf.int32)
    return tf.gather(padded, tf.minimum(idx, max_src - 1), batch_dims=1), labels


def _shift(audio, labels, clip_samples: int, max_shift: int):
    batch = tf.shape(audio)[0]
    shifts = tf.random.uniform([batch], -max_shift, max_shift + 1, dtype=tf.int32)
    padded = tf.pad(audio, [[0, 0], [max_shift, max_shift]])
    idx = tf.range(clip_samples)[tf.newaxis, :] + max_shift - shifts[:, tf.newaxis]
    return tf.gather(padded, idx, batch_dims=1), labels


def _load_bank(directory: str, sample_rate: int, clip_len: int | None,
               bank_size: int, seed: int) -> tf.Tensor | None:
    """Charge une banque de WAV en tenseur [N, L] (RIR ou bruits).

    - RIR (`clip_len=None`) : tronquées/complétées à 0,5 s, normalisées en
      énergie (la convolution ne doit pas changer le niveau global).
    - Bruits : un crop aléatoire (déterministe par seed) de `clip_len`
      échantillons par fichier — la variété vient du tirage par batch.
    Renvoie None si le dossier est vide ou absent."""
    import random
    from pathlib import Path

    import numpy as np
    import soundfile as sf

    root = Path(directory)
    files = sorted(root.rglob("*.wav")) if root.exists() else []
    if not files:
        return None
    rng = random.Random(seed)
    if len(files) > bank_size:
        files = rng.sample(files, bank_size)
    target = clip_len or sample_rate // 2
    rows = []
    for f in files:
        try:
            audio, sr = sf.read(f, dtype="float32")
        except Exception:
            continue
        if sr != sample_rate or len(audio) == 0:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if clip_len is not None and len(audio) > target:      # bruit : crop aléatoire
            start = rng.randrange(0, len(audio) - target + 1)
            audio = audio[start:start + target]
        audio = audio[:target]
        audio = np.pad(audio, (0, target - len(audio)))
        if clip_len is None:                                  # RIR : énergie unitaire
            energy = float(np.sqrt(np.sum(audio ** 2)))
            if energy < 1e-6:
                continue
            audio = audio / energy
        rows.append(audio.astype(np.float32))
    return tf.constant(np.stack(rows)) if rows else None


def _reverb(audio, labels, rirs: tf.Tensor, prob: float):
    """Convolution FFT avec une RIR tirée par clip ; niveau crête préservé."""
    batch = tf.shape(audio)[0]
    n = tf.shape(audio)[1]
    idx = tf.random.uniform([batch], 0, tf.shape(rirs)[0], dtype=tf.int32)
    rir = tf.gather(rirs, idx)
    fft_len = 2 ** 15                       # 16000 + 8000 - 1 < 32768
    conv = tf.signal.irfft(tf.signal.rfft(audio, [fft_len]) * tf.signal.rfft(rir, [fft_len]),
                           [fft_len])[:, :n]
    peak = tf.reduce_max(tf.abs(audio), axis=1, keepdims=True)
    conv_peak = tf.reduce_max(tf.abs(conv), axis=1, keepdims=True) + 1e-9
    conv = conv * (peak / conv_peak)
    keep = tf.random.uniform([batch]) < prob
    return tf.where(keep[:, tf.newaxis], conv, audio), labels


def _noise_mix(audio, labels, bank: tf.Tensor, prob: float, snr_lo: float, snr_hi: float):
    """Ajoute un bruit de la banque à un SNR tiré uniformément dans la plage."""
    batch = tf.shape(audio)[0]
    idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
    noise = tf.gather(bank, idx)
    a_rms = tf.sqrt(tf.reduce_mean(audio ** 2, axis=1, keepdims=True)) + 1e-9
    n_rms = tf.sqrt(tf.reduce_mean(noise ** 2, axis=1, keepdims=True)) + 1e-9
    snr = tf.random.uniform([batch, 1], snr_lo, snr_hi)
    gain = a_rms / (n_rms * tf.pow(10.0, snr / 20.0))
    mixed = tf.clip_by_value(audio + gain * noise, -1.0, 1.0)
    keep = tf.random.uniform([batch]) < prob
    return tf.where(keep[:, tf.newaxis], mixed, audio), labels


def make_dataset(paths: list[str], labels: list[int], wakeword: WakewordConfig,
                 batch_size: int, shuffle: bool = False, augment: bool = False,
                 augmentation: AugmentationConfig | None = None,
                 seed: int = 42) -> tf.data.Dataset:
    n = wakeword.clip_samples
    features = FeatureExtractor(wakeword)

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels, tf.float32)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda path, label: _decode(path, label, n), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.cache()                       # le décodage WAV ne se refait pas à chaque epoch

    aug = augmentation or AugmentationConfig()
    if augment and aug.enabled:
        max_shift = int(aug.time_shift_ms / 1000 * wakeword.sample_rate)
        ds = ds.map(lambda a, lab: _speed(a, lab, n, aug.speed_min, aug.speed_max),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if max_shift > 0:
            ds = ds.map(lambda a, lab: _shift(a, lab, n, max_shift),
                        num_parallel_calls=tf.data.AUTOTUNE)
        if aug.rir_prob > 0:
            rirs = _load_bank(aug.rir_dir, wakeword.sample_rate, None,
                              aug.bank_size, seed)
            if rirs is None:
                raise RuntimeError(f"rir_prob={aug.rir_prob} mais aucune RIR 16 kHz "
                                   f"dans {aug.rir_dir} (cf. docs/DATA.md)")
            ds = ds.map(lambda a, lab: _reverb(a, lab, rirs, aug.rir_prob),
                        num_parallel_calls=tf.data.AUTOTUNE)
        if aug.noise_prob > 0:
            bank = _load_bank(aug.noise_dir, wakeword.sample_rate, n,
                              aug.bank_size, seed)
            if bank is None:
                raise RuntimeError(f"noise_prob={aug.noise_prob} mais aucun bruit "
                                   f"16 kHz dans {aug.noise_dir}")
            lo, hi = aug.noise_snr_db
            ds = ds.map(lambda a, lab: _noise_mix(a, lab, bank, aug.noise_prob, lo, hi),
                        num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda a, lab: (features(a), lab), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
