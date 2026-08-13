"""Point d'entrée unique : `coachvocal <domaine> <action>`.

Tout ce que fait le projet passe par ici — préparer les données, entraîner,
évaluer, promouvoir, écouter, servir. Aucun script à éditer pour changer un
paramètre : `--set training.epochs=10` suffit, et la commande exacte se retrouve
dans les artefacts du run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import paths, registry
from .config import list_experiments, load_experiment, load_wakeword

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Coach vocal — pipeline wake word de bout en bout.")
data_app = typer.Typer(no_args_is_help=True, help="Données : split, dataset, audit, TTS.")
reg_app = typer.Typer(no_args_is_help=True, help="Registre de modèles et promotions.")
live_app = typer.Typer(no_args_is_help=True, help="Inférence au micro.")
app.add_typer(data_app, name="data")
app.add_typer(reg_app, name="registry")
app.add_typer(live_app, name="live")

console = Console()
SET_HELP = "Surcharge ponctuelle, ex. --set training.epochs=5 (répétable)"


# ── Config ────────────────────────────────────────────────────────────────────
@app.command("experiments")
def experiments_cmd():
    """Liste les expériences disponibles."""
    for name in list_experiments():
        cfg = load_experiment(name)
        console.print(f"[bold]{name}[/] — {cfg.description or 'sans description'}")
        console.print(f"    mot-clé {cfg.wakeword.name} · dataset {cfg.dataset.name} · "
                      f"modèle {cfg.model.arch} · seeds {cfg.training.seeds}")


@app.command("config")
def config_cmd(experiment: str, set_: list[str] = typer.Option(None, "--set", help=SET_HELP)):
    """Affiche la configuration résolue (après composition et surcharges)."""
    console.print_json(json.dumps(load_experiment(experiment, set_).model_dump(), default=str))


# ── Données ───────────────────────────────────────────────────────────────────
@data_app.command("split")
def data_split(wakeword: str, seed: int = 42, force: bool = False):
    """Fige `splits.csv` (par groupe). Refuse d'écraser un split existant."""
    from .data import splits as splits_mod

    out = splits_mod.freeze(paths.word_dir(wakeword), wakeword, seed=seed, force=force)
    console.print(f"✅  {out}")
    console.print_json(json.dumps(splits_mod.summarize(out)))


@data_app.command("build")
def data_build(experiment: str, set_: list[str] = typer.Option(None, "--set", help=SET_HELP),
               out: Optional[Path] = None):
    """Construit le manifest du dataset (sans entraîner) et vérifie les fuites."""
    from .data import build

    cfg = load_experiment(experiment, set_)
    manifest = build(cfg)
    target = out or paths.report_dir("datasets") / f"{cfg.dataset.name}_manifest.csv"
    manifest.to_csv(target)
    console.print(f"\n💾  {target}")


@data_app.command("audit")
def data_audit(experiment: str, set_: list[str] = typer.Option(None, "--set", help=SET_HELP)):
    """Audit qualité du dataset (+ PNG) — à lancer avant tout entraînement."""
    from .data import build
    from .data.quality import audit, plot

    cfg = load_experiment(experiment, set_)
    manifest = build(cfg)
    rep = audit(manifest, cfg.wakeword.sample_rate, cfg.wakeword.audio.clip_seconds,
                seed=cfg.dataset.data_seed)
    out = paths.report_dir("data_quality")
    (out / f"{cfg.dataset.name}.json").parent.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg.dataset.name}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    png = plot(rep, out / f"{cfg.dataset.name}.png", f"Dataset {cfg.dataset.name}")
    for issue in rep["issues"]:
        console.print(f"  ⚠️  {issue}")
    console.print(f"{'✅  aucun problème' if rep['ok'] else '⚠️  problèmes détectés'} — {png}")


@data_app.command("gate")
def data_gate(experiment: str, set_: list[str] = typer.Option(None, "--set", help=SET_HELP)):
    """Porte qualité (ADR-007) : mesure chaque clip de la recette et le classe
    accepté / rejeté / douteux. Les douteux s'auditent dans l'interface
    (page Qualité), puis `quality_gate.enabled: true` filtre le build."""
    from .data import sources as src_registry
    from .data.gate import gate_dir, run_gate

    cfg = load_experiment(experiment, set_)
    gate_cfg = cfg.dataset.quality_gate
    ctx = src_registry.SourceContext(wakeword=cfg.wakeword, dataset=cfg.dataset)
    files: dict[str, list] = {}
    for source_cfg in cfg.dataset.enabled_sources():
        if any(tag in source_cfg.name for tag in gate_cfg.skip_pools):
            continue                       # silencieux par conception
        pools = src_registry.get(source_cfg.type)(source_cfg, ctx)
        files[source_cfg.name] = sorted({f for s in source_cfg.splits
                                         for f in pools.get(s, [])})
    report = run_gate(files, gate_cfg, cfg.wakeword.sample_rate, cfg.wakeword.name)
    n_doubt = report["counts"]["douteux"]
    if n_doubt:
        console.print(f"👂  {n_doubt} douteux à auditer : make ui → page Qualité")
    console.print(f"💾  {gate_dir(cfg.wakeword.name)}")


@data_app.command("tts-pool")
def data_tts_pool(wakeword: str, per_combo: Optional[int] = None):
    """Génère le pool de positifs synthétiques Piper décrit dans la config du mot."""
    from .data.tts import generate_pool, piper_available

    word = load_wakeword(wakeword)
    if word.tts is None:
        raise typer.BadParameter(f"aucune section `tts` dans configs/wakeword/{wakeword}.yaml")
    if not piper_available():
        raise typer.BadParameter("`uvx` introuvable — installer uv (cf. docs/DATA.md)")
    generate_pool(
        paths.word_dir(wakeword), word.tts.text,
        [v.model_dump() for v in word.tts.voices], word.tts.length_scales,
        word.tts.noise_scales, per_combo or word.tts.per_combo,
        word.sample_rate, word.clip_samples, word.tts.pool_name)


@data_app.command("sources")
def data_sources():
    """Liste les types de sources disponibles pour une recette."""
    from .data import sources as src

    console.print("\n".join(f"  {name}" for name in src.available()))


# ── Entraînement ──────────────────────────────────────────────────────────────
@app.command("train")
def train_cmd(experiment: str,
              run_id: Optional[str] = typer.Option(None, help="Identifiant du run (défaut : nom de l'expérience)"),
              set_: list[str] = typer.Option(None, "--set", help=SET_HELP),
              track: bool = typer.Option(True, help="Suivi MLflow"),
              skip_audit: bool = typer.Option(False, help="Sauter l'audit qualité")):
    """Entraîne les candidats, élit par la validation, produit les preuves."""
    from .training import train

    cfg = load_experiment(experiment, set_)
    summary = train(cfg, run_id=run_id, track=track, skip_audit=skip_audit)
    console.print(f"\n📁  {paths.run_dir(cfg.wakeword.name, run_id or cfg.name)}")
    console.print("ℹ️   Promouvoir si le banc streaming le confirme : "
                  f"coachvocal registry promote {run_id or cfg.name} --reason \"…\"")
    return summary


# ── Banc streaming ────────────────────────────────────────────────────────────
@app.command("bench")
def bench_cmd(wakeword: str = "eloquence",
              runs: list[str] = typer.Option(None, "--run", help="Runs à comparer (défaut : champion)"),
              minutes: float = typer.Option(4.0, help="Durée totale d'audio continu"),
              thresholds: str = typer.Option("0.5,0.8", help="Seuils à mesurer")):
    """Banc streaming sur audio continu — la mesure qui décide des promotions."""
    from .evaluation import stream_bench

    word = load_wakeword(wakeword)
    runs = runs or [registry.champion_run(wakeword) or "v03"]
    models = {r: paths.run_dir(wakeword, r) / "model.keras" for r in runs}
    missing = [r for r, p in models.items() if not p.exists()]
    if missing:
        raise typer.BadParameter(f"modèle(s) introuvable(s) : {missing}")

    payload = stream_bench.run(
        models, word, minutes=minutes,
        thresholds=tuple(float(t) for t in thresholds.split(",")),
        splits_csv=paths.word_dir(wakeword) / "splits.csv")

    out = paths.report_dir("stream_bench") / f"{wakeword}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    console.print(f"\n💾  {out}")


# ── Registre ──────────────────────────────────────────────────────────────────
@reg_app.command("list")
def registry_list(wakeword: str = "eloquence"):
    """Tableau de tous les runs et de leurs métriques."""
    rows = registry.list_runs(wakeword)
    if not rows:
        console.print("aucun run — lancer `coachvocal train <experience>`")
        raise typer.Exit()
    table = Table(title=f"Runs — {wakeword}")
    for col in ("Run", "Date", "Seed", "F1", "FRR", "FAR", "AUC", "Champion"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["run"], (r["date"] or "")[:10], str(r["seed"]),
                      f"{r['f1']:.4f}" if r["f1"] else "—",
                      f"{r['frr']:.2%}" if r["frr"] is not None else "—",
                      f"{r['far']:.2%}" if r["far"] is not None else "—",
                      f"{r['roc_auc']:.4f}" if r["roc_auc"] else "—",
                      "⭐" if r["is_champion"] else "")
    console.print(table)


@reg_app.command("promote")
def registry_promote(run_id: str, wakeword: str = "eloquence",
                     reason: str = typer.Option(..., help="Justification CHIFFRÉE de la promotion")):
    """Promeut un run au rang de champion (met à jour `current/`)."""
    reg = registry.promote(wakeword, run_id, reason)
    console.print(f"⭐  {run_id} promu champion pour « {wakeword} »")
    console.print(f"    {reg['champion']['reason']}")


@reg_app.command("show")
def registry_show(wakeword: str = "eloquence"):
    """Champion courant et historique des promotions."""
    reg = registry.load(wakeword)
    if not reg.get("champion"):
        console.print("aucun champion promu")
        raise typer.Exit()
    c = reg["champion"]
    console.print(f"⭐  [bold]{c['run']}[/] (depuis {c['promoted']})\n    {c['reason']}")
    for h in reg.get("history", []):
        console.print(f"    ↳ {h['run']} ({h['promoted']} → {h.get('retired', '?')}) : {h['reason']}")


# ── Inférence ─────────────────────────────────────────────────────────────────
@live_app.command("devices")
def live_devices():
    """Liste les entrées audio disponibles."""
    from .inference.live import list_devices

    for d in list_devices():
        console.print(f"  [{d['index']:2}] {d['name']} ({d['channels']} ch)")


@live_app.command("listen")
def live_listen(wakeword: str = "eloquence", run: Optional[str] = None,
                threshold: Optional[float] = None, device: Optional[int] = None,
                save_triggers: bool = typer.Option(False, help="Sauver les buffers déclenchés")):
    """Détection always-on au micro avec le champion (ou `--run`)."""
    from . import runtime
    from .inference.live import stream

    runtime.configure(use_gpu=False)            # cf. ADR-002 : Metal fausse les probas
    word = load_wakeword(wakeword)
    model = registry.model_path(wakeword, run)
    console.print(f"📦  {model}")
    stream(model, word, device=device, threshold=threshold,
           save_dir=paths.word_dir(wakeword) / "trigger_clips" if save_triggers else None)


@live_app.command("guided")
def live_guided(wakeword: str = "eloquence", run: Optional[str] = None,
                threshold: Optional[float] = None, device: Optional[int] = None):
    """Test guidé essai par essai (produit des clips étiquetés)."""
    from . import runtime
    from .inference.live import guided

    runtime.configure(use_gpu=False)
    word = load_wakeword(wakeword)
    guided(registry.model_path(wakeword, run), word,
           paths.word_dir(wakeword) / "guided_clips", device=device, threshold=threshold)


# ── Services ──────────────────────────────────────────────────────────────────
@app.command("serve")
def serve_cmd(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Lance l'API FastAPI (Swagger sur /docs)."""
    import uvicorn

    uvicorn.run("coachvocal.serving.api:api", host=host, port=port, reload=reload)


@app.command("ui")
def ui_cmd(port: int = 8501):
    """Lance l'interface Streamlit."""
    import subprocess

    subprocess.run(["streamlit", "run", str(paths.ROOT / "app" / "Home.py"),
                    "--server.port", str(port)], check=False)


@app.command("dashboard")
def dashboard_cmd(wakeword: str = "eloquence"):
    """Génère le dashboard HTML comparatif (preuve archivable)."""
    from .evaluation.dashboard import build_dashboard

    out = build_dashboard(wakeword)
    console.print(f"💾  {out}")


if __name__ == "__main__":
    app()
