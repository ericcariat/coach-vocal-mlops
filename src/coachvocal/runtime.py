"""Réglages runtime TensorFlow (à appeler AVANT toute opération TF).

⚠️ Décision structurante du projet, cf. `docs/decisions/ADR-002` :
**l'entraînement se fait sur CPU**. Le plugin `tensorflow-metal` corrompt les
gradients sur cette machine — la loss explose au bout de quelques batches, avec
des hyperparamètres pourtant sains. Des heures ont été perdues à chercher un
bug d'apprentissage qui était un bug de backend. L'inférence sur Metal, elle,
est correcte et reste autorisée.
"""

from __future__ import annotations

import os
import random

import numpy as np


def configure(use_gpu: bool = False, seed: int | None = None) -> None:
    import tensorflow as tf

    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


def describe() -> dict:
    import tensorflow as tf

    return {
        "tensorflow": tf.__version__,
        "gpus_visibles": [d.name for d in tf.config.get_visible_devices("GPU")],
        "cpus": len(tf.config.list_physical_devices("CPU")),
    }
