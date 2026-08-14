"""Réglages runtime TensorFlow (à appeler AVANT toute opération TF).

⚠️ Décision structurante du projet, cf. `docs/decisions/ADR-002` :
**tout se fait sur CPU, entraînement ET inférence**. Le plugin
`tensorflow-metal` corrompt les gradients sur cette machine — la loss explose
au bout de quelques batches, avec des hyperparamètres pourtant sains — et
fausse aussi les probabilités à l'inférence (banc du 2026-07-28 : rappel
streaming 80 % sur CPU, 0 % sur Metal, même modèle et même audio). Des heures
ont été perdues à chercher un bug d'apprentissage qui était un bug de backend.
"""

from __future__ import annotations

import os
import random

import numpy as np


def configure(use_gpu: bool = False, seed: int | None = None) -> None:
    import tensorflow as tf

    if not use_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # TF déjà initialisé (autre appel plus tôt dans le processus) : la
            # liste ne peut plus changer. Acceptable UNIQUEMENT si aucun GPU
            # n'est déjà visible — sinon on refuse de continuer (ADR-002).
            if tf.config.get_visible_devices("GPU"):
                raise
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
