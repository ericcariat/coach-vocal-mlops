"""Registre des sources de données.

Ajouter une source = écrire un module ici et le décorer avec `@source("nom")`.
La recette YAML n'a alors qu'à mentionner `type: nom` — aucun `if` à ajouter
ailleurs dans le code. C'est ce qui rend le pipeline extensible sans le modifier
(nouveau mot-clé, nouveau corpus de négatifs, nouveau moteur TTS…).

Contrat d'une source : `fn(src: SourceConfig, ctx: SourceContext) -> dict[split, list[Path]]`
Elle DOIT être idempotente et respecter le split (jamais de fuite train↔test).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ... import paths
from ...config import DatasetConfig, SourceConfig, WakewordConfig

SplitPools = dict[str, list[Path]]
SourceFn = Callable[["SourceConfig", "SourceContext"], SplitPools]

_REGISTRY: dict[str, SourceFn] = {}


def source(name: str):
    def deco(fn: SourceFn) -> SourceFn:
        if name in _REGISTRY:
            raise ValueError(f"source '{name}' déjà enregistrée")
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> SourceFn:
    if name not in _REGISTRY:
        raise KeyError(f"source inconnue '{name}' — disponibles : {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class SourceContext:
    """Tout ce dont une source a besoin, sans qu'elle connaisse le reste du monde."""

    wakeword: WakewordConfig
    dataset: DatasetConfig
    _splits_cache: dict | None = field(default=None, repr=False)

    @property
    def sr(self) -> int:
        return self.wakeword.sample_rate

    @property
    def clip_samples(self) -> int:
        return self.wakeword.clip_samples

    @property
    def seed(self) -> int:
        return self.dataset.data_seed

    @property
    def word_dir(self) -> Path:
        return paths.word_dir(self.wakeword.name)

    def cache(self, name: str) -> Path:
        return paths.cache_dir(self.wakeword.name, name)

    def word_splits(self) -> dict[str, dict[str, list[Path]]]:
        """`splits.csv` → {split: {label: [chemins]}}.

        Le split est FIGÉ une fois pour toutes, par groupe (vidéo source /
        locuteur) et jamais re-tiré : c'est la garantie anti-fuite du projet.
        """
        if self._splits_cache is not None:
            return self._splits_cache
        csv_path = self.word_dir / self.dataset.splits_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} absent — le split doit être figé avant tout entraînement "
                "(coachvocal data split)"
            )
        out: dict = {s: defaultdict(list) for s in ("train", "val", "test")}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                sub = "positives" if row["label"] == self.wakeword.name else "negatives_proches"
                out[row["split"]][row["label"]].append(self.word_dir / "clean" / sub / row["file"])
        self._splits_cache = out
        return out


def _done(root: Path) -> Path:
    return root / ".done"


def cached(root: Path) -> SplitPools | None:
    """Renvoie le pool si le cache est complet, sinon None.

    Les pools dérivés (crops MUSAN, décodages Common Voice, TTS…) sont
    régénérables : ils vivent dans artifacts/cache/ et ne sont pas versionnés.
    Le drapeau `.done` évite de reprendre une génération interrompue à moitié.
    """
    if not _done(root).exists():
        return None
    return {s: sorted((root / s).glob("*.wav")) for s in ("train", "val", "test")}


def mark_done(root: Path) -> SplitPools:
    _done(root).touch()
    return {s: sorted((root / s).glob("*.wav")) for s in ("train", "val", "test")}


# Import des modules concrets → remplit le registre (effet de bord assumé).
from . import (  # noqa: E402,F401
    common_voice,
    fragments,
    gsc,
    guided,
    musan,
    silence,
    speech_negatives,
    tts_piper,
    word_clips,
)
