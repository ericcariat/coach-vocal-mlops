"""Activations custom sérialisables.

`relu_max` est mathématiquement identique à ReLU, mais son graphe passe par
`tf.maximum` au lieu de l'op `Relu` — ce qui échappe au noyau fusionné
MatMul+BiasAdd+ReLU défectueux de tensorflow-metal (ADR-002, § Re-contrôles).

Pour un export embarqué (TFLite), reconstruire l'architecture avec `relu`
standard et y charger les mêmes poids : fonction strictement identique, et le
convertisseur retrouve la fusion Conv/Dense+ReLU optimisée.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="coachvocal")
def relu_max(x):
    return tf.maximum(x, 0.0)
