"""DS-CNN — convolutions séparables en profondeur (Zhang et al., « Hello Edge », 2017).

Référence du domaine pour la détection de mot-clé embarquée : à précision égale
elle utilise 5 à 10 fois moins de paramètres et de MACs qu'une CNN classique,
parce qu'elle factorise chaque convolution en un filtrage spatial par canal
suivi d'un mélange 1×1. C'est le candidat naturel si le détecteur doit un jour
tourner en permanence sans vider la batterie.

Comparaison attendue face à `cnn_baseline` : F1 équivalente, empreinte divisée.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from . import architecture


def _ds_block(x, filters: int, stride: int, idx: int):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                               name=f"dw{idx}")(x)
    x = layers.BatchNormalization(name=f"dw_bn{idx}")(x)
    x = layers.Activation("relu", name=f"dw_relu{idx}")(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"pw{idx}")(x)
    x = layers.BatchNormalization(name=f"pw_bn{idx}")(x)
    return layers.Activation("relu", name=f"pw_relu{idx}")(x)


@architecture("dscnn")
def dscnn(input_shape, filters=64, n_blocks=4, dropout=0.2, name="dscnn"):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters, (10, 4), strides=(2, 1), padding="same", use_bias=False,
                      name="stem")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    for i in range(n_blocks):
        x = _ds_block(x, filters, stride=1, idx=i + 1)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out, name=name)
