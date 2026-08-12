"""CNN binaire de référence — l'architecture de tous les runs v01 → v03.

Deux convolutions, un pooling, deux denses. Volontairement simple : à ce
volume de données (quelques milliers de clips), la capacité n'est pas le
facteur limitant — la qualité et la diversité des négatifs le sont (cf. le banc
streaming). Elle sert de témoin face à toute nouvelle architecture.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from . import architecture
from .activations import relu_max

# Les chaînes Keras standard passent telles quelles ; seules les activations
# custom du projet ont besoin d'être résolues en fonction.
_CUSTOM_ACTIVATIONS = {"relu_max": relu_max}


@architecture("cnn_baseline")
def cnn_baseline(input_shape, filters=(32, 64), dense_units=128,
                 dropout_conv=0.25, dropout_dense=0.5, activation="relu",
                 name="cnn_baseline"):
    # `activation` est paramétrable car le noyau ReLU *fusionné* de
    # tensorflow-metal est défectueux sur les couches Dense (ADR-002,
    # re-contrôle du 2026-08-12) : "leaky_relu" et "relu_max" échappent
    # à la fusion.
    activation = _CUSTOM_ACTIVATIONS.get(activation, activation)
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(filters[0], 3, activation=activation, padding="same", name="conv1"),
            layers.Conv2D(filters[1], 3, activation=activation, padding="same", name="conv2"),
            layers.MaxPooling2D(2, name="pool"),
            layers.Dropout(dropout_conv, name="drop1"),
            layers.Flatten(name="flatten"),
            layers.Dense(dense_units, activation=activation, name="dense1"),
            layers.Dropout(dropout_dense, name="drop2"),
            layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name=name,
    )


@architecture("cnn_norm")
def cnn_norm(input_shape, filters=(32, 64, 64), dense_units=128, dropout=0.3,
             name="cnn_norm"):
    """Variante avec BatchNorm et pooling global.

    Le pooling global remplace le `Flatten` : beaucoup moins de paramètres
    denses, donc moins de sur-apprentissage sur les rares positifs réels, et une
    tolérance accrue au décalage temporel du mot dans la fenêtre."""
    model = keras.Sequential([layers.Input(shape=input_shape)], name=name)
    for i, f in enumerate(filters):
        model.add(layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i + 1}"))
        model.add(layers.BatchNormalization(name=f"bn{i + 1}"))
        model.add(layers.Activation("relu", name=f"relu{i + 1}"))
        model.add(layers.MaxPooling2D(2, name=f"pool{i + 1}"))
    model.add(layers.GlobalAveragePooling2D(name="gap"))
    model.add(layers.Dropout(dropout, name="drop"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense1"))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))
    return model
