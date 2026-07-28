"""Front-end acoustique partagé entre entraînement et inférence."""

from .features import FeatureExtractor, log_mel, sliding_windows

__all__ = ["FeatureExtractor", "log_mel", "sliding_windows"]
