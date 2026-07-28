"""Restitution : figures PNG + rapport Markdown.

Règle du projet : **tout résultat produit une preuve regardable par un humain**.
Un JSON de métriques ne se conteste pas à l'œil ; une courbe d'apprentissage,
une matrice de confusion et un taux de déclenchement par pool, si.

Palette vérifiée pour les déficiences de la vision des couleurs (deutan/protan)
et lisible en clair comme en sombre.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE = "#5B6FB8"
ORANGE = "#C2452C"
GREEN = "#00806B"
GRAY = "#6E7076"


def learning_curve(history: dict, out: Path, title: str = "") -> Path:
    ep = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Courbes d'apprentissage — {title}", fontsize=12)
    for ax, key, label in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(ep, history[key], "o-", color=ORANGE, lw=2, label="Train")
        if f"val_{key}" in history:
            ax.plot(ep, history[f"val_{key}"], "s-", color=BLUE, lw=2, label="Validation")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    return _save(fig, out)


def confusion_png(cm: np.ndarray, classes: list[str], out: Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — {title}")
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save(fig, out)


def threshold_png(sweep: list[dict], out: Path, live_threshold: float | None = None) -> Path:
    """FRR et FAR en fonction du seuil : le compromis se CHOISIT, il ne se subit
    pas. Pour un always-on on privilégie la FAR basse (moins de réveils
    intempestifs), quitte à répéter le mot de temps en temps."""
    th = [s["threshold"] for s in sweep]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(th, [s["frr"] * 100 for s in sweep], "o-", color=ORANGE, lw=2,
            label="FRR — mot raté")
    ax.plot(th, [s["far"] * 100 for s in sweep], "s-", color=BLUE, lw=2,
            label="FAR — fausse alarme")
    if live_threshold is not None:
        ax.axvline(live_threshold, color=GRAY, ls="--", lw=1.5)
        ax.text(live_threshold, ax.get_ylim()[1] * 0.92, " seuil live", color=GRAY, fontsize=9)
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("%")
    ax.set_title("Compromis FRR / FAR selon le seuil (test par clips)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def pools_png(breakdown: dict, out: Path) -> Path:
    pools = sorted(breakdown, key=lambda p: breakdown[p]["fire_rate"])
    rates = [breakdown[p]["fire_rate"] * 100 for p in pools]
    colors = [GREEN if breakdown[p]["ok"] else ORANGE for p in pools]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(pools))))
    ax.barh(pools, rates, color=colors)
    ax.set_xlabel("% de clips au-dessus du seuil")
    ax.set_title("Taux de déclenchement par pool (test)\nvert = conforme à l'attendu")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def write_run_report(run_dir: Path, cfg, metrics: dict, manifest, extra: list[str] | None = None) -> Path:
    """`report.md` : ce qu'un humain lit avant de décider quoi que ce soit."""
    test = metrics["test"]
    lines = [
        f"# Run `{run_dir.name}` — expérience `{cfg.name}`",
        "",
        cfg.description or "",
        "",
        "## Configuration",
        "",
        f"- mot-clé : **{cfg.wakeword.name}** — {cfg.wakeword.sample_rate} Hz, "
        f"{cfg.wakeword.audio.num_mel_bins} mels, fenêtre {cfg.wakeword.audio.clip_seconds} s",
        f"- dataset : **{cfg.dataset.name}** (seed données {cfg.dataset.data_seed}, "
        f"empreinte `{manifest.fingerprint()}`)",
        f"- modèle : **{cfg.model.name}** (`{cfg.model.arch}`)",
        f"- entraînement : {metrics.get('epochs_run', '?')}/{cfg.training.epochs} epochs, "
        f"batch {cfg.training.batch_size}, lr {cfg.training.learning_rate}, "
        f"seeds {cfg.training.seeds} (élu par `{cfg.training.selection_metric}`)",
        "",
        "## Composition du jeu de données",
        "",
        "| Split | Positifs | Négatifs | Ratio |",
        "|---|---:|---:|---|",
    ]
    for split, c in manifest.balance().items():
        pos, neg = c.get("pos", 0), c.get("neg", 0)
        lines.append(f"| {split} | {pos} | {neg} | 1:{neg / max(pos, 1):.1f} |")

    lines += [
        "",
        f"## Test par clips (seuil {test['threshold']})",
        "",
        "| Métrique | Valeur |",
        "|---|---:|",
        f"| Accuracy | {test['accuracy']:.2%} |",
        f"| F1 (classe positive) | {test['f1_pos']:.4f} |",
        f"| **FRR** (mot raté) | {test['frr']:.2%} |",
        f"| **FAR** (fausse alarme) | {test['far']:.2%} |",
        f"| ROC-AUC | {test['roc_auc']:.4f} |",
        f"| Clips évalués | {test['n']} (dont {test['n_pos']} positifs) |",
        "",
        "> Rappel : ces chiffres décrivent des clips de 1 s pré-découpés. La",
        "> décision de promotion se prend au **banc streaming**",
        "> (`coachvocal bench`), qui reproduit les conditions réelles.",
        "",
        "## Preuves",
        "",
        "`learning_curve.png` · `confusion.png` · `threshold.png` · `pools.png`",
    ]
    if extra:
        lines += ["", *extra]
    out = run_dir / "report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def save_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return path
