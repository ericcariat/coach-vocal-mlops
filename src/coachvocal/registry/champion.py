"""Registre des modèles : quel modèle est en production, et pourquoi.

Sans ce fichier, « le bon modèle » est une information qui vit dans la tête de
celui qui a lancé le dernier entraînement — et le script live pointe un chemin
codé en dur qui devient faux à chaque run. Ici :

- `CHAMPION.json` contient le run promu, la DATE, la RAISON chiffrée et
  l'historique complet des promotions ;
- le lien `current/` pointe le dossier du champion — tout le code d'inférence
  (live, API, UI) charge `current/model.keras` et rien d'autre.

Une promotion est un acte tracé, pas un `cp`.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from .. import paths


def registry_file(wakeword: str) -> Path:
    return paths.RUNS / wakeword / "CHAMPION.json"


def current_link(wakeword: str) -> Path:
    return paths.RUNS / wakeword / "current"


def load(wakeword: str) -> dict:
    f = registry_file(wakeword)
    return json.loads(f.read_text()) if f.exists() else {"champion": None, "history": []}


def champion_run(wakeword: str) -> str | None:
    champion = load(wakeword).get("champion")
    return champion.get("run") if champion else None


def model_path(wakeword: str, run_id: str | None = None) -> Path:
    """Chemin du modèle à charger. Sans `run_id` : le champion courant."""
    if run_id:
        return paths.run_dir(wakeword, run_id) / "model.keras"
    link = current_link(wakeword)
    if link.exists():
        return link / "model.keras"
    reg = load(wakeword)
    if reg.get("champion"):
        return paths.run_dir(wakeword, reg["champion"]["run"]) / "model.keras"
    raise FileNotFoundError(
        f"aucun champion pour « {wakeword} » — promouvoir un run : "
        f"coachvocal registry promote <run_id> --reason ...")


def promote(wakeword: str, run_id: str, reason: str, evidence: dict | None = None) -> dict:
    """Promeut un run. `reason` doit contenir des CHIFFRES : c'est la trace qui
    permettra six mois plus tard de savoir sur quoi la décision reposait."""
    run = paths.run_dir(wakeword, run_id)
    if not (run / "model.keras").exists():
        raise FileNotFoundError(f"{run / 'model.keras'} introuvable")

    reg = load(wakeword)
    entry = {"run": run_id, "promoted": date.today().isoformat(), "reason": reason,
             "evidence": evidence or {}}
    if reg.get("champion"):
        previous = dict(reg["champion"])
        previous["retired"] = entry["promoted"]
        reg["history"] = [previous, *reg.get("history", [])]
    reg["champion"] = entry

    registry_file(wakeword).parent.mkdir(parents=True, exist_ok=True)
    registry_file(wakeword).write_text(json.dumps(reg, indent=2, ensure_ascii=False))

    link = current_link(wakeword)
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(run.resolve(), link, target_is_directory=True)
    return reg


def list_runs(wakeword: str) -> list[dict]:
    """Tous les runs avec leurs métriques — matière première du dashboard."""
    root = paths.RUNS / wakeword
    if not root.exists():
        return []
    champion = load(wakeword).get("champion") or {}
    out = []
    for d in sorted(root.iterdir()):
        metrics_file = d / "metrics.json"
        if not d.is_dir() or d.name == "current" or not metrics_file.exists():
            continue
        m = json.loads(metrics_file.read_text())
        test = m.get("test", {})
        out.append({
            "run": d.name,
            "date": m.get("date", ""),
            "experiment": m.get("experiment", ""),
            "seed": m.get("selected_seed"),
            "accuracy": test.get("accuracy"),
            "f1": test.get("f1_pos"),
            "frr": test.get("frr"),
            "far": test.get("far"),
            "roc_auc": test.get("roc_auc"),
            "dataset_fingerprint": m.get("dataset_fingerprint"),
            "is_champion": d.name == champion.get("run"),
            "path": str(d),
        })
    return out


def bench_results(wakeword: str) -> dict:
    """Derniers résultats du banc streaming, s'ils existent."""
    f = paths.report_dir("stream_bench") / f"{wakeword}.json"
    return json.loads(f.read_text()) if f.exists() else {}
