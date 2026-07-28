"""Registre d'architectures.

Comparer des architectures est une exigence du projet ; il faut donc pouvoir en
ajouter une sans toucher au reste. Une archi = une fonction décorée
`@architecture("nom")` qui renvoie un `keras.Model` NON compilé (la compilation
appartient à l'entraîneur, qui connaît le learning rate).
"""

from __future__ import annotations

from collections.abc import Callable

ArchFn = Callable[..., "object"]
_REGISTRY: dict[str, ArchFn] = {}


def architecture(name: str):
    def deco(fn: ArchFn) -> ArchFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def build(arch: str, input_shape: tuple[int, ...], **params):
    if arch not in _REGISTRY:
        raise KeyError(f"architecture inconnue '{arch}' — disponibles : {sorted(_REGISTRY)}")
    return _REGISTRY[arch](input_shape=input_shape, **params)


def available() -> list[str]:
    return sorted(_REGISTRY)


from . import cnn, dscnn  # noqa: E402,F401
