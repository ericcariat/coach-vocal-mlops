"""Recette YAML → manifest concret.

Une seule fonction publique : `build(cfg)`. Elle instancie chaque source du
dataset, vérifie l'absence de fuite entre splits, et produit le manifest.
Aucune connaissance des sources en dur : tout passe par le registre.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..config import ExperimentConfig
from . import sources as src_registry
from .manifest import Manifest


def build(cfg: ExperimentConfig, verbose: bool = True) -> Manifest:
    ctx = src_registry.SourceContext(wakeword=cfg.wakeword, dataset=cfg.dataset)
    manifest = Manifest()

    if verbose:
        print(f"=== Dataset « {cfg.dataset.name} » (seed données {cfg.dataset.data_seed}) ===")

    gate_cfg = cfg.dataset.quality_gate
    excluded: set[str] | None = None
    if gate_cfg.enabled:
        from .gate import load_exclusions
        excluded = load_exclusions(cfg.wakeword.name, gate_cfg.doubt_policy)
        if excluded is None:
            raise RuntimeError(
                "quality_gate.enabled=true mais aucune porte n'a tourné : lancer "
                "`coachvocal data gate <experiment>` d'abord (le build ne filtre "
                "jamais en silence).")
        if verbose:
            print(f"  porte qualité active : {len(excluded)} clip(s) exclus")

    for source_cfg in cfg.dataset.enabled_sources():
        fn = src_registry.get(source_cfg.type)
        pools = fn(source_cfg, ctx)
        skip = any(tag in source_cfg.name for tag in gate_cfg.skip_pools)
        got = {}
        for split in source_cfg.splits:
            files = pools.get(split, [])
            if excluded is not None and not skip:
                files = [f for f in files if Path(f).name not in excluded]
            got[split] = len(files)
            if files:
                manifest.add(source_cfg.name, files, source_cfg.label, split, source_cfg.copies)
        if verbose:
            boost = f" ×{source_cfg.copies}" if source_cfg.copies > 1 else ""
            print(f"  {source_cfg.name:<18} {source_cfg.type:<18} label={source_cfg.label}"
                  f"{boost:<4} {got}")

    check_leakage(manifest)
    if verbose:
        print("\n=== Composition (après boosts) ===")
        for line in manifest.summary_lines():
            print(line)
        print(f"  empreinte dataset : {manifest.fingerprint()}")
    return manifest


def check_leakage(manifest: Manifest) -> None:
    """Aucun fichier ne doit apparaître dans deux splits différents.

    Échoue fort et tôt : une fuite train/test invalide toutes les métriques du
    run, et c'est la première chose qu'un jury vérifie."""
    seen: dict[str, str] = {}
    leaks: dict[str, set] = defaultdict(set)
    for r in manifest.rows:
        key = Path(r["file"]).name
        if key in seen and seen[key] != r["split"]:
            leaks[key] |= {seen[key], r["split"]}
        seen[key] = r["split"]
    if leaks:
        detail = ", ".join(f"{k} ({'/'.join(sorted(v))})" for k, v in list(leaks.items())[:5])
        raise RuntimeError(
            f"FUITE DE DONNÉES : {len(leaks)} fichier(s) présents dans plusieurs splits — {detail}")
