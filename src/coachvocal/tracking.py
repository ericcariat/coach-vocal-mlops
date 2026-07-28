"""Suivi d'expériences (MLflow), en écriture locale.

Pourquoi MLflow plutôt qu'un dossier de JSON : la comparaison. Avec 5 candidats
par run, plusieurs recettes de dataset et plusieurs architectures, la question
« qu'est-ce qui a changé entre ce run et celui-là ? » doit se répondre en
triant un tableau, pas en ouvrant des fichiers.

Le wrapper est volontairement mince et **tolérant** : si MLflow est absent
(image Docker d'inférence, CI), l'entraînement continue et log dans le vide.
Les artefacts du run restent la source de vérité sur disque.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import paths


def _flatten(prefix: str, obj: Any, out: dict) -> dict:
    """Aplatit une config imbriquée en `a.b.c = valeur` (MLflow ne prend que
    des params scalaires)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(obj, ensure_ascii=False)[:490]
    else:
        out[prefix] = obj
    return out


class Tracker:
    """Interface unique ; ne lève jamais pour un problème de tracking."""

    def __init__(self, experiment: str, enabled: bool = True):
        self.enabled = enabled
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            print("ℹ️  mlflow absent — suivi désactivé (artefacts disque inchangés)")
            self.enabled = False
            return
        paths.MLRUNS.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(paths.mlflow_uri())
        mlflow.set_experiment(experiment)
        self.mlflow = mlflow

    @contextmanager
    def run(self, name: str, tags: dict | None = None, nested: bool = False):
        if not self.enabled:
            yield self
            return
        with self.mlflow.start_run(run_name=name, nested=nested):
            if tags:
                self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
            yield self

    def log_config(self, cfg_dict: dict) -> None:
        if not self.enabled:
            return
        params = _flatten("", cfg_dict, {})
        # MLflow limite à 100 params par appel
        items = list(params.items())
        for i in range(0, len(items), 90):
            self.mlflow.log_params(dict(items[i:i + 90]))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        clean = {k.replace("/", "_"): float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float))}
        if clean:
            self.mlflow.log_metrics(clean, step=step)

    def log_history(self, history: dict) -> None:
        if not self.enabled:
            return
        for epoch in range(len(next(iter(history.values()), []))):
            self.log_metrics({k: v[epoch] for k, v in history.items()}, step=epoch)

    def log_artifacts(self, directory: Path) -> None:
        if not self.enabled or not Path(directory).exists():
            return
        self.mlflow.log_artifacts(str(directory))

    def log_model(self, model_path: Path) -> None:
        if not self.enabled or not Path(model_path).exists():
            return
        self.mlflow.log_artifact(str(model_path), artifact_path="model")
