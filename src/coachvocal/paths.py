"""Emplacements canoniques du projet.

Tout le code passe par ici : aucun chemin n'est écrit en dur ailleurs. C'est
ce qui permet de déplacer le projet, de le monter dans Docker (`/app`) ou de
pointer un disque externe sans toucher une ligne de logique.
"""

from __future__ import annotations

import os
from pathlib import Path

# Racine du dépôt = deux niveaux au-dessus de src/coachvocal/, sauf override.
ROOT = Path(os.environ.get("COACHVOCAL_ROOT", Path(__file__).resolve().parents[2]))

CONFIGS = ROOT / "configs"

# ── Données (hors git, versionnées par DVC) ───────────────────────────────────
DATA = Path(os.environ.get("COACHVOCAL_DATA", ROOT / "data"))
EXTERNAL = DATA / "external"          # datasets téléchargés, immuables (MUSAN, CV, GSC)
WAKEWORDS = DATA / "wakewords"        # un sous-dossier par mot-clé

# ── Artefacts produits (hors git sauf métriques/rapports) ─────────────────────
ARTIFACTS = Path(os.environ.get("COACHVOCAL_ARTIFACTS", ROOT / "artifacts"))
RUNS = ARTIFACTS / "runs"             # un entraînement = un dossier
REPORTS = ARTIFACTS / "reports"       # preuves PNG/HTML (cf. règle du projet)
CACHE = ARTIFACTS / "cache"           # pools dérivés régénérables
MLRUNS = ARTIFACTS / "mlruns"         # backend MLflow local

DOCS = ROOT / "docs"


def word_dir(wakeword: str) -> Path:
    """Dossier de données d'un mot-clé (raw/, clean/, generated/, splits.csv…)."""
    return WAKEWORDS / wakeword


def run_dir(wakeword: str, run_id: str) -> Path:
    return RUNS / wakeword / run_id


def cache_dir(wakeword: str, source: str) -> Path:
    """Dossier de cache d'un pool dérivé (régénérable, jamais dans git)."""
    return CACHE / wakeword / source


def report_dir(name: str) -> Path:
    return REPORTS / name


def mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"file://{MLRUNS}")


def ensure_dirs() -> None:
    for d in (EXTERNAL, WAKEWORDS, RUNS, REPORTS, CACHE, MLRUNS):
        d.mkdir(parents=True, exist_ok=True)
