"""Couche données : recette YAML → manifest → tf.data."""

from .builder import build, check_leakage
from .manifest import Manifest

__all__ = ["build", "check_leakage", "Manifest"]
