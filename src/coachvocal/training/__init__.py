"""Entraînement : construction des tf.data et protocole multi-candidats."""

from .datasets import make_dataset
from .trainer import train

__all__ = ["make_dataset", "train"]
