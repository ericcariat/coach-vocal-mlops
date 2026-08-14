"""Page Évaluation — toutes les métriques de tous les runs, et les preuves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import paths, registry  # noqa: E402

st.set_page_config(page_title="Évaluation", page_icon="📈", layout="wide")
st.title("Évaluation")

WAKEWORD = "eloquence"
runs = registry.list_runs(WAKEWORD)
if not runs:
    st.info("Aucun run. Lancer un entraînement depuis la page précédente.")
    st.stop()

# ── Le champion d'abord : ses chiffres officiels (banc) en tête de page ──────
champ_name = registry.champion_run(WAKEWORD)
if champ_name:
    try:
        cm = json.loads((paths.run_dir(WAKEWORD, champ_name) / "metrics.json").read_text())
        if cm.get("bench"):
            th_live = f"th{cm.get('live_threshold', 0.8)}"
            r = cm["bench"].get(th_live) or next(iter(cm["bench"].values()))
            st.success(f"**Champion : `{champ_name}`** — banc streaming au seuil live "
                       f"{th_live[2:]} : **{r['recall_stream']:.1%}** de rappel "
                       f"({r['detected']}/{r['n_occ']}) · **{r['fa_per_hour']:.1f} FA/h** · "
                       f"mots proches {cm.get('cousins', {}).get('moi_@0.8', 0):.0%}")
        else:
            t = cm.get("test", {})
            st.success(f"**Champion : `{champ_name}`** — F1 {t.get('f1_pos', 0):.4f} · "
                       f"FRR {t.get('frr', 0):.2%} · FAR {t.get('far', 0):.2%}")
    except Exception:
        pass

st.subheader("Comparatif — toutes les métriques")
df = pd.DataFrame([{
    "Run": r["run"] + (" ⭐" if r["is_champion"] else ""),
    "Date": (r["date"] or "")[:10], "Seed": r["seed"],
    "Accuracy": r["accuracy"], "F1": r["f1"], "FRR ↓": r["frr"], "FAR ↓": r["far"],
    "Empreinte data": r["dataset_fingerprint"],
} for r in runs])
st.dataframe(df.style.format({"Accuracy": "{:.2%}", "F1": "{:.4f}", "FRR ↓": "{:.2%}",
                              "FAR ↓": "{:.2%}"}, na_rep="—"),
             width="stretch", hide_index=True)

st.caption("**FRR** (*False Rejection Rate*) : le mot est prononcé, rien ne se passe. "
           "**FAR** (*False Acceptance Rate*) : un négatif déclenche.")

st.error("⚠️ Ces chiffres décrivent des clips d'1 s pré-découpés. Ils ont déjà **mal "
         "classé** les modèles — la décision de promotion se prend au **banc streaming**.")

# ── Rappel vs FA/h au banc : toutes les versions sur un même graphique ────────
st.subheader("Rappel vs FA/h au banc — toutes les versions comparables")
st.caption("Un point = un modèle à un seuil. En haut à gauche = l'idéal.")

bench_dir = paths.report_dir("stream_bench")
archives = sorted(p for p in bench_dir.glob(f"{WAKEWORD}_2*.json")
                   if "verdicts" not in p.name)
compositions: dict = {}
for p in archives:
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    sig = (round(d.get("total_seconds", 0)), d.get("n_occurrences"),
           tuple(sorted(Path(s["wav"]).name for s in d.get("segments", []))))
    compositions.setdefault(sig, {"date": d.get("date", ""), "points": {}})
    compositions[sig]["date"] = max(compositions[sig]["date"], d.get("date", ""))
    DK, LK = "d'", "l'"
    for model, per in d.get("results", {}).items():
        for th, r in per.items():
            fb = r.get("recall_by_form", {})

            def forme(k, _fb=fb):
                return (f"{_fb[k]['detected']}/{_fb[k]['n_occ']}" if k in _fb else "—")

            compositions[sig]["points"][(model, th)] = {
                "Modèle": model, "Seuil": float(th[2:]),
                "Rappel": round(r["recall_stream"] * 100, 1),
                "FA/h": round(r["fa_per_hour"], 1),
                "Détections": f"{r['detected']}/{r['n_occ']}",
                "FA": r["false_alarms"],
                "d'": forme(DK), "l'": forme(LK),
                "Incertains": r["uncertain"],
            }

if compositions:
    ordered = sorted(compositions.items(), key=lambda kv: kv[1]["date"], reverse=True)
    labels = {f"{v['date'][:16]} — {k[1]} occurrences · {k[0] // 60} min · "
              f"{len(v['points'])} points": k for k, v in ordered}
    choice = st.selectbox("Composition de banc", list(labels))
    sig = labels[choice]
    pts = pd.DataFrame(list(compositions[sig]["points"].values()))

    tous_modeles = sorted(pts["Modèle"].unique())
    champion = registry.champion_run(WAKEWORD)
    # Au banc, le champion peut apparaître sous son nom de fichier de tête
    # (ex. eloquence_frab30np_64x3_seed46.onnx pour le run oww_frab30np_s46) :
    # le run déclare cet alias dans metrics.json → `bench_model`.
    champ_labels = {champion}
    try:
        mfile = paths.run_dir(WAKEWORD, champion) / "metrics.json"
        alias = json.loads(mfile.read_text()).get("bench_model")
        if alias:
            champ_labels.add(alias)
    except Exception:
        pass
    # défaut : le champion + les modèles maison récents (pas les 15 d'un coup)
    defaut = [m for m in tous_modeles if m in champ_labels or m.startswith(("v1",))][-6:]
    for c in champ_labels:
        if c in tous_modeles and c not in defaut:
            defaut.append(c)
    sel = st.multiselect("Modèles affichés", tous_modeles, default=defaut or tous_modeles[:5])
    aff = pts[pts["Modèle"].isin(sel)]

    # Le graphique du comparatif HTML, embarqué tel quel : lignes par modèle
    # (seuil croissant = FA décroissantes), seuil étiqueté sous chaque point,
    # infobulle au survol.
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#8a6fc9", "#7a7a2c"]
    couleurs = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(tous_modeles)}
    data_js = json.dumps([{**r, "c": couleurs[r["Modèle"]]}
                          for r in aff.to_dict("records")], ensure_ascii=False)
    xmax = max(70, float(aff["FA/h"].max()) * 1.1) if len(aff) else 70
    html = """
<div style="font:13px ui-sans-serif,-apple-system,sans-serif;color:#23211C;background:#FBFAF7;border-radius:12px;padding:12px 14px">
<svg id="ch" viewBox="0 0 900 440"
 style="width:100%;max-width:900px;height:auto;display:block"></svg>
<div id="tip" style="position:fixed;pointer-events:none;background:#23211C;
 color:#FBFAF7;font-size:12px;line-height:1.5;padding:7px 10px;border-radius:6px;
 opacity:0;max-width:250px"></div>
<div id="leg" style="display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:4px"></div>
</div>
<script>
const DATA=""" + data_js + """;
const XMAX=""" + str(round(xmax, 1)) + """;
const NS="http://www.w3.org/2000/svg", svg=document.getElementById("ch");
const X0=52,Y0=16,W=830,H=340,YMIN=0,YMAX=105;
const px=f=>X0+f/XMAX*W, py=r=>Y0+(YMAX-r)/(YMAX-YMIN)*H;
function el(t,a,x){const n=document.createElementNS(NS,t);
 for(const k in a)n.setAttribute(k,a[k]);if(x!==undefined)n.textContent=x;
 svg.appendChild(n);return n;}
for(let f=0;f<=XMAX;f+=XMAX>200?100:10){
 el("line",{x1:px(f),y1:Y0,x2:px(f),y2:Y0+H,stroke:"#D8D3C8","stroke-dasharray":"2 5"});
 el("text",{x:px(f),y:Y0+H+22,"text-anchor":"middle","font-size":11,fill:"#6E685C"},f);}
for(let r=0;r<=100;r+=20){
 el("line",{x1:X0,y1:py(r),x2:X0+W,y2:py(r),stroke:"#D8D3C8","stroke-dasharray":"2 5"});
 el("text",{x:X0-8,y:py(r)+4,"text-anchor":"end","font-size":11,fill:"#6E685C"},r+"%");}
el("line",{x1:X0,y1:Y0+H,x2:X0+W,y2:Y0+H,stroke:"#23211C","stroke-width":1.5});
el("text",{x:X0+W/2,y:Y0+H+48,"text-anchor":"middle","font-size":12,
 fill:"#6E685C"},
 "Fausses alarmes par heure (FA/h)  →  mieux = à gauche");
el("text",{x:14,y:Y0+H/2,"font-size":12,fill:"#6E685C",
 transform:"rotate(-90 14 "+(Y0+H/2)+")","text-anchor":"middle"},"rappel →");
const parMod={};
for(const d of DATA)(parMod[d["Modèle"]]=parMod[d["Modèle"]]||[]).push(d);
const tip=document.getElementById("tip");
for(const m in parMod){
 const pl=parMod[m].sort((a,b)=>a["Seuil"]-b["Seuil"]);
 if(pl.length>1)el("polyline",{points:pl.map(d=>px(d["FA/h"])+","+py(d["Rappel"])).join(" "),
  fill:"none",stroke:pl[0].c,"stroke-width":2,opacity:.5});
 for(const d of pl){
  el("circle",{cx:px(d["FA/h"]),cy:py(d["Rappel"]),r:6.5,fill:d.c,
   stroke:"#FBFAF7","stroke-width":2});
  el("text",{x:px(d["FA/h"]),y:py(d["Rappel"])+19,"text-anchor":"middle",
   "font-size":9.5,fill:"#6E685C"},d["Seuil"]);
  const h=el("circle",{cx:px(d["FA/h"]),cy:py(d["Rappel"]),r:15,fill:"transparent"});
  h.addEventListener("mousemove",ev=>{tip.style.opacity=1;
   tip.style.left=(ev.clientX+13)+"px";tip.style.top=(ev.clientY-8)+"px";
   tip.innerHTML="<b>"+d["Modèle"]+"</b> · seuil "+d["Seuil"]+"<br>rappel "+
    d["Rappel"]+"% ("+d["Détections"]+") · "+d["FA/h"]+" FA/h<br>d' "+d["d'"]+
    " · l' "+d["l'"]+" · incertains "+d["Incertains"];});
  h.addEventListener("mouseleave",()=>tip.style.opacity=0);}
 // étiquette directe au point le plus haut du modèle
 const top=pl.reduce((a,b)=>b["Rappel"]>a["Rappel"]?b:a);
 el("text",{x:px(top["FA/h"])+10,y:py(top["Rappel"])-9,"font-size":11.5,
  "font-weight":650,fill:top.c},m);}
const leg=document.getElementById("leg");
for(const m in parMod){const s=document.createElement("span");
 s.style.cssText="display:inline-flex;align-items:center;gap:5px;font-size:12px;color:#6E685C";
 s.innerHTML='<span style="width:11px;height:11px;border-radius:3px;background:'+
  parMod[m][0].c+'"></span>'+m;leg.appendChild(s);}
</script>"""
    import streamlit.components.v1 as components
    components.html(html, height=560)

    st.markdown("**Le tableau complet (modèles × seuils, avec les formes) :**")
    st.dataframe(pts.sort_values(["Rappel", "FA/h"], ascending=[False, True]),
                 width="stretch", hide_index=True)
else:
    st.info("Aucune archive de banc — lancer `make bench MINUTES=60`.")

# ── Détail d'un run ───────────────────────────────────────────────────────────
run = st.selectbox("Détail du run", [r["run"] for r in runs])
run_dir = paths.run_dir(WAKEWORD, run)
metrics = json.loads((run_dir / "metrics.json").read_text())

test = metrics.get("test", {})
if test:
    c = st.columns(4)
    c[0].metric("F1", f"{test.get('f1_pos', 0):.4f}")
    c[1].metric("FRR", f"{test.get('frr', 0):.2%}")
    c[2].metric("FAR", f"{test.get('far', 0):.2%}")
    c[3].metric("Seed élue", metrics.get("selected_seed", "—"))
elif metrics.get("bench"):
    # Tête openWakeWord (ADR-008) : pas de test par clips — ses métriques sont
    # celles du banc streaming, la mesure qui décide de toute façon.
    b = metrics["bench"]
    th_live = str(metrics.get("live_threshold", 0.8))
    cols = st.columns(len(b) + 1)
    for i, (th, r) in enumerate(sorted(b.items())):
        seuil = th.replace("th", "")
        cols[i].metric(f"Banc @ {seuil}" + (" (live)" if seuil == th_live else ""),
                       f"{r['recall_stream']:.1%}",
                       f"{r['fa_per_hour']:.1f} FA/h", delta_color="off")
    cols[-1].metric("Seed élue", metrics.get("selected_seed", "—"))
    st.caption(f"Front-end : {metrics.get('frontend', '—')} · mots proches : "
               f"{metrics.get('cousins', {}).get('moi_@0.8', '—'):.0%} de déclenchement "
               f"@0.8 · archive `{metrics.get('bench_archive', '—')}`")

if metrics.get("candidates"):
    st.subheader("Candidats (protocole anti-variance)")
    st.dataframe(pd.DataFrame(metrics["candidates"]), width="stretch", hide_index=True)
    st.caption("L'élection se fait sur `val_loss`. Les colonnes de test sont là pour "
               "l'audit : élire dessus serait un biais de sélection.")

st.subheader("Preuves")
figures = [("learning_curve.png", "Courbes d'apprentissage"),
           ("confusion.png", "Matrice de confusion"),
           ("threshold.png", "Compromis FRR / FAR selon le seuil"),
           ("pools.png", "Taux de déclenchement par pool")]
cols = st.columns(2)
for i, (fname, label) in enumerate(figures):
    f = run_dir / fname
    if f.exists():
        cols[i % 2].image(str(f), caption=label)

if (run_dir / "report.md").exists():
    with st.expander("Rapport complet (report.md)"):
        st.markdown((run_dir / "report.md").read_text())
