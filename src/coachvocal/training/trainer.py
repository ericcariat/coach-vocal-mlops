"""Entraînement d'une expérience, du YAML aux preuves.

Protocole **multi-candidats** (leçon des runs v02/v03) : sur CPU, deux
entraînements à seed identique ne donnent pas le même modèle (ordonnancement des
threads, non-associativité des flottants, EarlyStopping qui coupe entre l'epoch
8 et 20). Variance mesurée : ±0.03 à 0.06 de F1 — soit plus que la plupart des
« améliorations » qu'on cherche à mesurer. Une conclusion a déjà dû être
rétractée pour cette raison.

Parade : entraîner N candidats (seeds différentes), **élire par la validation**,
ne regarder le test qu'une fois, à la fin. Élire par le test serait un biais de
sélection : on choisirait le modèle le plus chanceux sur le jeu qui sert à
prouver qu'il est bon.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import paths, runtime
from ..audio.features import FeatureExtractor
from ..config import ExperimentConfig
from ..data import build as build_manifest
from ..data.manifest import Manifest
from ..data.quality import audit as quality_audit
from ..evaluation import clip_eval, report
from ..models import build as build_model
from ..tracking import Tracker
from .datasets import make_dataset


def _class_weight(labels: list[int]) -> dict:
    """Rééquilibrage par la loss. Les négatifs sont 20 à 40 fois plus nombreux :
    sans pondération, prédire toujours « non » donne 97 % d'accuracy — d'où
    l'inutilité de l'accuracy sur ce problème."""
    n = len(labels)
    n_pos = int(sum(labels))
    n_neg = n - n_pos
    return {0: n / (2 * max(n_neg, 1)), 1: n / (2 * max(n_pos, 1))}


def _fit_candidate(cfg: ExperimentConfig, manifest: Manifest, seed: int, out_dir: Path,
                   tracker: Tracker) -> dict:
    from tensorflow import keras

    runtime.configure(use_gpu=cfg.training.use_gpu, seed=seed)

    tr_paths, tr_labels = manifest.paths_labels("train")
    va_paths, va_labels = manifest.paths_labels("val")
    te_paths, te_labels = manifest.paths_labels("test")

    train_ds = make_dataset(tr_paths, tr_labels, cfg.wakeword, cfg.training.batch_size,
                            shuffle=True, augment=True,
                            augmentation=cfg.dataset.augmentation, seed=seed)
    val_ds = make_dataset(va_paths, va_labels, cfg.wakeword, cfg.training.batch_size)
    test_ds = make_dataset(te_paths, te_labels, cfg.wakeword, cfg.training.batch_size)

    features = FeatureExtractor(cfg.wakeword)
    model = build_model(cfg.model.arch, features.input_shape, **cfg.model.params)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
                  loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

    fit_started = datetime.now()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.training.epochs,
        class_weight=_class_weight(tr_labels) if cfg.training.class_weight else None,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.training.early_stopping_patience,
            restore_best_weights=True)],
        verbose=2).history
    fit_s = round((datetime.now() - fit_started).total_seconds(), 1)

    y_proba = model.predict(test_ds, verbose=0).ravel()
    y_true = np.concatenate([lab.numpy() for _, lab in test_ds]).ravel().astype(int)
    test_metrics = clip_eval.evaluate(y_true, y_proba, cfg.training.threshold)

    # Rappel sur les positifs de VAL (contrainte de l'élection fa_ambient) —
    # jamais le test : le test ne sert qu'une fois, à la fin.
    va_proba = model.predict(val_ds, verbose=0).ravel()
    va_true = np.concatenate([lab.numpy() for _, lab in val_ds]).ravel().astype(int)
    pos = va_true == 1
    val_recall = float((va_proba[pos] >= cfg.training.threshold).mean()) if pos.any() else 0.0

    fa_ambient = None
    if cfg.training.selection_metric == "fa_ambient":
        fa_ambient = _ambient_fa_per_hour(model, cfg)

    best_ep = int(np.argmin(history["val_loss"]))
    candidate = {
        "seed": seed,
        "fit_s": fit_s,
        "val_recall": val_recall,
        "fa_ambient": fa_ambient,
        "epochs_run": len(history["loss"]),
        "best_epoch": best_ep + 1,
        "val_loss": float(history["val_loss"][best_ep]),
        "val_accuracy": float(history["val_accuracy"][best_ep]),
        "test": test_metrics,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "y_proba": [float(p) for p in y_proba],
        "y_true": [int(t) for t in y_true],
        "test_files": te_paths,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2))

    tracker.log_history(history)
    tracker.log_metrics({"val_loss_best": candidate["val_loss"], "fit_s": fit_s,
                         **{f"test_{k}": v for k, v in test_metrics.items()
                            if isinstance(v, (int, float))}})
    return candidate


def _ambient_fa_per_hour(model, cfg: ExperimentConfig) -> float | None:
    """FA/h du candidat sur le flux ambiant de validation (sans occurrence du
    mot, vérifié) — la MÊME machine à états que le live et le banc.
    None si le dossier val_ambient est vide."""
    import soundfile as sf

    from ..inference.detector import WakeWordDetector

    root = paths.word_dir(cfg.wakeword.name) / "val_ambient"
    wavs = sorted(root.glob("*.wav")) if root.exists() else []
    if not wavs:
        return None
    detector = WakeWordDetector(model, cfg.wakeword)
    triggers, seconds = 0, 0.0
    for wav in wavs:
        audio, sr = sf.read(wav, dtype="float32")
        if sr != cfg.wakeword.sample_rate:
            continue
        triggers += len(detector.run_offline(audio))
        seconds += len(audio) / sr
    return round(triggers / (seconds / 3600), 2) if seconds else None


def _elect(candidates: list[dict], training) -> tuple[dict, str]:
    """Élection par la VALIDATION (jamais le test). Renvoie (élu, motif).

    - val_loss / val_accuracy : historique (min loss / max accuracy) ;
    - fa_ambient : formulation PRODUIT (microWakeWord, LiveKit) — parmi les
      candidats à rappel val ≥ contrainte, celui aux FA/h ambiantes minimales.
      Si aucun ne satisfait la contrainte : le meilleur rappel val (et on le
      dit), plutôt qu'un modèle sourd champion du silence."""
    metric = training.selection_metric
    if metric == "fa_ambient":
        with_fa = [c for c in candidates if c.get("fa_ambient") is not None]
        if not with_fa:
            raise RuntimeError("selection_metric=fa_ambient mais aucun candidat "
                               "n'a de FA/h ambiantes (val_ambient/ vide ?)")
        ok = [c for c in with_fa if c["val_recall"] >= training.selection_min_val_recall]
        if ok:
            best = min(ok, key=lambda c: c["fa_ambient"])
            return best, (f"min FA/h ambiantes ({best['fa_ambient']}/h) parmi "
                          f"{len(ok)} candidat(s) à rappel val ≥ "
                          f"{training.selection_min_val_recall:.0%}")
        best = max(with_fa, key=lambda c: c["val_recall"])
        return best, (f"AUCUN candidat n'atteint le rappel val "
                      f"{training.selection_min_val_recall:.0%} — repli sur le "
                      f"meilleur rappel ({best['val_recall']:.1%})")
    if metric.replace("val_", "") == "loss":
        best = min(candidates, key=lambda c: c["val_loss"])
        return best, f"min(val_loss {best['val_loss']:.4f})"
    best = max(candidates, key=lambda c: c["val_accuracy"])
    return best, f"max(val_accuracy {best['val_accuracy']:.4f})"


def train(cfg: ExperimentConfig, run_id: str | None = None, track: bool = True,
          skip_audit: bool = False) -> dict:
    """Entraîne, sélectionne, produit les artefacts. Renvoie le résumé du run."""
    run_id = run_id or cfg.name
    run_dir = paths.run_dir(cfg.wakeword.name, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()

    manifest = build_manifest(cfg)
    manifest.to_csv(run_dir / "manifest.csv")

    quality = None
    if not skip_audit:
        print("\n=== Audit qualité ===")
        quality = quality_audit(manifest, cfg.wakeword.sample_rate,
                                cfg.wakeword.audio.clip_seconds, seed=cfg.dataset.data_seed)
        for issue in quality["issues"]:
            print(f"  ⚠️  {issue}")
        if quality["ok"]:
            print("  ✅ aucun problème détecté")
        report.save_json(quality, run_dir / "data_quality.json")

    tracker = Tracker(experiment=f"{cfg.wakeword.name}/{cfg.dataset.name}", enabled=track)
    cfg_dump = cfg.model_dump()

    candidates: list[dict] = []
    with tracker.run(run_id, tags={"wakeword": cfg.wakeword.name, "arch": cfg.model.arch,
                                   "dataset": cfg.dataset.name,
                                   "dataset_fingerprint": manifest.fingerprint()}):
        tracker.log_config(cfg_dump)
        for seed in cfg.training.seeds:
            print(f"\n{'=' * 62}\n🚀  Candidat seed={seed} "
                  f"({cfg.training.seeds.index(seed) + 1}/{len(cfg.training.seeds)})\n{'=' * 62}")
            with tracker.run(f"{run_id}-seed{seed}", tags={"seed": seed}, nested=True) as t:
                candidates.append(
                    _fit_candidate(cfg, manifest, seed, run_dir / "candidates" / f"seed{seed}", t))

        # ── Élection par la VALIDATION (jamais par le test) ───────────────────
        best, motif = _elect(candidates, cfg.training)
        print(f"\n🏅  Élu : seed {best['seed']} ({motif} — sélection par la validation)")

        summary = _finalize(cfg, run_id, run_dir, manifest, candidates, best, quality, started)
        tracker.log_metrics({f"selected_test_{k}": v for k, v in best["test"].items()
                             if isinstance(v, (int, float))})
        tracker.log_artifacts(run_dir)
    return summary


def _finalize(cfg, run_id, run_dir: Path, manifest: Manifest, candidates: list[dict],
              best: dict, quality: dict | None, started: datetime) -> dict:
    """Copie le modèle élu, génère métriques, figures et rapport.

    Les artefacts finaux sont RECONSTRUITS depuis les données stockées du
    candidat — on ne ré-entraîne pas pour les produire, puisque ce ne serait pas
    reproductible à l'identique."""
    import shutil

    src = run_dir / "candidates" / f"seed{best['seed']}" / "model.keras"
    shutil.copy2(src, run_dir / "model.keras")

    y_true = np.array(best["y_true"])
    y_proba = np.array(best["y_proba"])
    probas_by_file = dict(zip(best["test_files"], best["y_proba"]))
    breakdown = clip_eval.pool_breakdown(manifest, probas_by_file, "test", cfg.training.threshold)

    metrics = {
        "run_id": run_id,
        "experiment": cfg.name,
        "date": started.isoformat(timespec="seconds"),
        "duration_s": round((datetime.now() - started).total_seconds()),
        "selected_seed": best["seed"],
        "selection": f"min({cfg.training.selection_metric}) parmi {cfg.training.seeds}",
        "epochs_run": best["epochs_run"],
        "val_loss": best["val_loss"],
        "val_accuracy": best["val_accuracy"],
        "test": best["test"],
        "pools_test": breakdown,
        "dataset_fingerprint": manifest.fingerprint(),
        "fit_s_total": round(sum(c.get("fit_s", 0) for c in candidates), 1),
        "use_gpu": cfg.training.use_gpu,
        "candidates": [{"seed": c["seed"], "val_loss": c["val_loss"],
                        "val_recall": c.get("val_recall"),
                        "fa_ambient": c.get("fa_ambient"),
                        "test_f1": c["test"]["f1_pos"], "test_frr": c["test"]["frr"],
                        "test_far": c["test"]["far"], "epochs": c["epochs_run"],
                        "fit_s": c.get("fit_s")}
                       for c in candidates],
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        **runtime.describe()},
        "data_quality_ok": None if quality is None else quality["ok"],
    }
    report.save_json(metrics, run_dir / "metrics.json")
    report.save_json(cfg.model_dump(), run_dir / "config.json")

    title = f"{cfg.wakeword.name} · {run_id} · seed {best['seed']}"
    report.learning_curve(best["history"], run_dir / "learning_curve.png", title)
    report.confusion_png(clip_eval.confusion(y_true, y_proba, cfg.training.threshold),
                         cfg.wakeword.classes, run_dir / "confusion.png", title)
    report.threshold_png(best["test"]["threshold_sweep"], run_dir / "threshold.png",
                         cfg.wakeword.live.threshold)
    report.pools_png(breakdown, run_dir / "pools.png")

    extra = ["## Candidats (tous les seeds)", "",
             "| Seed | val_loss | F1 test | FRR | FAR | Epochs |", "|---|---:|---:|---:|---:|---:|"]
    for c in metrics["candidates"]:
        star = " ⭐" if c["seed"] == best["seed"] else ""
        extra.append(f"| {c['seed']}{star} | {c['val_loss']:.4f} | {c['test_f1']:.4f} | "
                     f"{c['test_frr']:.2%} | {c['test_far']:.2%} | {c['epochs']} |")
    extra += ["", "> L'élection se fait sur `val_loss`. Les colonnes de test sont montrées",
              "> *a posteriori* pour l'audit : les utiliser pour choisir serait un biais",
              "> de sélection."]
    report.write_run_report(run_dir, cfg, metrics, manifest, extra)

    print(f"\n💾  Run complet : {run_dir}")
    print(f"    F1 {best['test']['f1_pos']:.4f} · FRR {best['test']['frr']:.2%} · "
          f"FAR {best['test']['far']:.2%} · AUC {best['test']['roc_auc']:.4f}")
    return metrics
