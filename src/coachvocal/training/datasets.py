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
        ds = ds.map(lambda a, lab: _shift(a, lab, n, max_shift),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda a, lab: (features(a), lab), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
