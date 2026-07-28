"""Registre de modèles : champion courant, historique des promotions."""

from .champion import (
    bench_results,
    champion_run,
    list_runs,
    load,
    model_path,
    promote,
)

__all__ = ["bench_results", "champion_run", "list_runs", "load", "model_path", "promote"]
