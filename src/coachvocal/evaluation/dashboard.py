"""Dashboard HTML comparatif — la preuve qu'un humain peut contester.

Réunit sur une page : tous les runs avec TOUTES leurs métriques par clip, les
résultats du banc streaming, et le champion courant avec sa justification. Page
autonome (aucune ressource externe), lisible en thème clair et sombre, palette
vérifiée pour les déficiences de vision des couleurs.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from .. import paths, registry

CSS = """
:root {
  --bg:#FBFBFC; --panel:#FFFFFF; --border:#E3E4E8; --text:#1B1C1F; --muted:#6E7076;
  --accent:#5B6FB8; --warn:#C2452C; --ok:#00806B; --star:#B8860B;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#141518; --panel:#1C1D21; --border:#303138; --text:#E9EAEE; --muted:#9A9CA4;
          --accent:#7286D3; --warn:#D96A3E; --ok:#2E9E85; --star:#D6A93B; }
}
:root[data-theme="light"] { --bg:#FBFBFC; --panel:#FFFFFF; --border:#E3E4E8; --text:#1B1C1F;
  --muted:#6E7076; --accent:#5B6FB8; --warn:#C2452C; --ok:#00806B; --star:#B8860B; }
:root[data-theme="dark"] { --bg:#141518; --panel:#1C1D21; --border:#303138; --text:#E9EAEE;
  --muted:#9A9CA4; --accent:#7286D3; --warn:#D96A3E; --ok:#2E9E85; --star:#D6A93B; }
body { margin:0; background:var(--bg); color:var(--text);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:2.5rem 1.25rem 4rem; }
h1 { font-size:1.75rem; margin:0 0 .25rem; letter-spacing:-.02em; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; }
.sub { color:var(--muted); margin:0 0 2rem; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:1rem 1.25rem; margin-bottom:1rem; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { padding:.55rem .7rem; text-align:right; border-bottom:1px solid var(--border);
  white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; font-size:12.5px; text-transform:uppercase;
  letter-spacing:.04em; }
tr.champion td { background:color-mix(in srgb, var(--star) 10%, transparent); font-weight:600; }
.note { color:var(--muted); font-size:13.5px; border-left:3px solid var(--accent);
  padding:.35rem 0 .35rem .85rem; margin:.75rem 0; }
.warn { border-left-color:var(--warn); }
.kpi { display:flex; gap:1rem; flex-wrap:wrap; }
.kpi div { flex:1 1 150px; }
.kpi .v { font-size:1.6rem; font-weight:650; letter-spacing:-.02em; }
.kpi .l { color:var(--muted); font-size:12.5px; text-transform:uppercase; letter-spacing:.04em; }
code { background:color-mix(in srgb, var(--accent) 12%, transparent); padding:.1rem .35rem;
  border-radius:4px; font-size:13px; }
"""


def _fmt(v, kind="num"):
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "f4":
        return f"{v:.4f}"
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def _runs_table(rows: list[dict]) -> str:
    head = ("Run", "Date", "Seed", "Accuracy", "F1", "FRR ↓", "FAR ↓", "ROC-AUC", "Empreinte data")
    out = ["<div class='scroll'><table><thead><tr>"]
    out += [f"<th>{h}</th>" for h in head]
    out.append("</tr></thead><tbody>")
    for r in rows:
        cls = " class='champion'" if r["is_champion"] else ""
        star = " ⭐" if r["is_champion"] else ""
        out.append(
            f"<tr{cls}><td>{html.escape(r['run'])}{star}</td><td>{r['date'][:10]}</td>"
            f"<td>{r['seed']}</td><td>{_fmt(r['accuracy'], 'pct')}</td>"
            f"<td>{_fmt(r['f1'], 'f4')}</td><td>{_fmt(r['frr'], 'pct')}</td>"
            f"<td>{_fmt(r['far'], 'pct')}</td><td>{_fmt(r['roc_auc'], 'f4')}</td>"
            f"<td><code>{r['dataset_fingerprint'] or '—'}</code></td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _bench_table(bench: dict) -> str:
    if not bench.get("results"):
        return "<p class='note warn'>Aucun banc streaming enregistré — lancer <code>coachvocal bench</code>.</p>"
    thresholds = sorted({k for m in bench["results"].values() for k in m})
    out = ["<div class='scroll'><table><thead><tr><th>Modèle</th>"]
    for th in thresholds:
        out.append(f"<th>Rappel {th[2:]}</th><th>FA/h {th[2:]}</th><th>Incertains</th>")
    out.append("</tr></thead><tbody>")
    for name, per_th in bench["results"].items():
        out.append(f"<tr><td>{html.escape(name)}</td>")
        for th in thresholds:
            r = per_th.get(th, {})
            out.append(f"<td>{_fmt(r.get('recall_stream'), 'pct')}</td>"
                       f"<td>{_fmt(r.get('fa_per_hour'))}</td>"
                       f"<td>{r.get('uncertain', '—')}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def build_dashboard(wakeword: str, out_path: Path | None = None) -> Path:
    rows = registry.list_runs(wakeword)
    bench = registry.bench_results(wakeword)
    reg = registry.load(wakeword)
    champion = reg.get("champion") or {}

    kpi = ""
    if bench.get("results") and champion.get("run") in bench["results"]:
        best = bench["results"][champion["run"]]
        th = "th0.8" if "th0.8" in best else next(iter(best))
        r = best[th]
        kpi = f"""<div class='panel kpi'>
          <div><div class='v'>{r['recall_stream']:.1%}</div><div class='l'>Rappel streaming</div></div>
          <div><div class='v'>{r['fa_per_hour']:.1f}</div><div class='l'>Fausses alarmes / heure</div></div>
          <div><div class='v'>{r['n_occ']}</div><div class='l'>Occurrences testées</div></div>
          <div><div class='v'>{bench['total_seconds'] / 60:.0f} min</div><div class='l'>Audio continu</div></div>
        </div>"""

    champion_block = (
        f"<p class='note'>⭐ Champion : <b>{html.escape(champion.get('run', '—'))}</b> "
        f"(promu le {champion.get('promoted', '—')})<br>{html.escape(champion.get('reason', ''))}</p>"
        if champion else "<p class='note warn'>Aucun champion promu.</p>")

    body = f"""<div class="wrap">
<h1>Wake word « {html.escape(wakeword)} » — tableau de bord</h1>
<p class="sub">Généré le {datetime.now():%Y-%m-%d %H:%M} · toutes les métriques de tous les runs</p>
{champion_block}
{kpi}

<h2>Banc streaming — la mesure qui décide</h2>
<p class="note">Logique live rejouée sur de l'audio YouTube continu jamais vu à
l'entraînement, vérité terrain WhisperX. <b>Rappel</b> = occurrences réellement
attrapées ; <b>FA/h</b> = réveils intempestifs par heure. Corpus thématique donc
pire cas volontaire. Les événements « incertains » (vérité terrain douteuse) sont
exclus du décompte.</p>
{_bench_table(bench)}

<h2>Test par clips — signal de contrôle</h2>
<p class="note warn">⚠️ Ces métriques ont déjà <b>mal classé</b> les modèles : le
meilleur en F1 par clip s'est révélé le pire en conditions réelles. À lire comme
un garde-fou, jamais comme un critère de promotion.
<b>FRR</b> = mot prononcé et raté ; <b>FAR</b> = négatif accepté.</p>
{_runs_table(rows)}

<h2>Comment reproduire</h2>
<div class="panel">
<p><code>coachvocal train &lt;experience&gt;</code> → entraîne les candidats et élit par la validation<br>
<code>coachvocal bench --run &lt;run&gt; --minutes 16</code> → mesure en conditions réelles<br>
<code>coachvocal registry promote &lt;run&gt; --reason "…"</code> → promotion tracée</p>
<p class="sub">Empreinte data : deux runs de même empreinte ont vu exactement les
mêmes fichiers avec les mêmes poids — un écart de métriques sans écart d'empreinte
est de la variance, pas un progrès.</p>
</div>
</div>"""

    doc = (f"<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Wake word {html.escape(wakeword)} — dashboard</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")

    out = out_path or paths.report_dir("dashboards") / f"{wakeword}_{datetime.now():%Y-%m-%d}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


def export_json(wakeword: str) -> dict:
    """Même contenu, exploitable par l'API et l'UI."""
    return {"runs": registry.list_runs(wakeword),
            "champion": registry.load(wakeword).get("champion"),
            "bench": registry.bench_results(wakeword)}


if __name__ == "__main__":
    print(json.dumps(export_json("eloquence"), indent=2, default=str))
