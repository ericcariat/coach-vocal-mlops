"""Re-contrôle de la cause racine d'ADR-002 : le ReLU fusionné sur Metal.

Le plugin `tensorflow-metal` (encore en 1.2.0, constaté le 2026-08-12) fusionne
MatMul + BiasAdd + ReLU en un seul noyau GPU… qui renvoie la sortie SANS
appliquer le ReLU. Les négatifs traversent le réseau entier : gradients faux à
l'entraînement, probabilités fausses à l'inférence. Références :
- https://developer.apple.com/forums/thread/818015
- https://github.com/tensorflow/tensorflow/issues/62137 (même bug, TF 2.14)

Usage : `uv run python scripts/check_metal_relu.py`
Sortie attendue le jour où Apple aura corrigé : trois `0.0` et « corrigé ».
Tant qu'un min est négatif, ADR-002 tient — ne pas réactiver `use_gpu`.
"""

import numpy as np
import tensorflow as tf

weights = [np.ones((10, 5), np.float32) * -1, np.ones(5, np.float32) * -1]
data = np.ones((1, 10), np.float32)


def dense_relu_min(separate: bool) -> float:
    layers = [tf.keras.layers.Input(shape=(10,))]
    if separate:
        layers += [tf.keras.layers.Dense(5), tf.keras.layers.Activation("relu")]
    else:
        layers += [tf.keras.layers.Dense(5, activation="relu")]
    model = tf.keras.Sequential(layers)
    model.layers[0].set_weights(weights)
    return float(model.predict(data, verbose=0).min())


if not tf.config.get_visible_devices("GPU"):
    raise SystemExit("Aucun GPU visible : ce contrôle doit tourner sur Metal.")

mins = {
    "Dense(activation='relu') fusionné": dense_relu_min(separate=False),
    "Dense + Activation('relu') séparé": dense_relu_min(separate=True),
    "tf.nn.relu direct (hors graphe)": float(tf.nn.relu(tf.constant([-3.0, 2.0])).numpy().min()),
}
for name, m in mins.items():
    print(f"  {name} : min = {m}")

if any(m < 0 for m in mins.values()):
    print("\n❌ Bug toujours présent : le ReLU fusionné laisse passer les négatifs.")
    print("   ADR-002 tient — entraînement et inférence restent sur CPU.")
    raise SystemExit(1)
print("\n✅ Corrigé : relancer les deux tests de non-régression d'ADR-002 avant "
      "d'envisager use_gpu: true.")
