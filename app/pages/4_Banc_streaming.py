"""Page Banc streaming — la mesure qui décide, et l'audit des erreurs à l'oreille."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import registry  # noqa: E402

st.set_page_config(page_title="Banc streaming", page_icon="🎬", layout="wide")
st.title("🎬 Banc streaming")

WAKEWORD = "eloquence"
bench = registry.bench_results(WAKEWORD)

st.markdown("""
On rejoue la **logique live exacte** (fenêtre d'1 s, décision toutes les 125 ms,
3 fenêtres consécutives, cooldown) sur de l'audio YouTube **continu jamais vu à
l'entraînement**, avec pour vérité terrain les alignements WhisperX.

Deux chiffres décrivent l'expérience réelle : le **rappel streaming** (occurrences
réellement attrapées) et les **fausses alarmes par heure**.
""")

if not bench.get("results"):
    st.info("Aucun banc enregistré.")
    st.code("uv run coachvocal bench --run v03_replica --minutes 16", language="bash")
    st.stop()

c = st.columns(3)
c[0].metric("Audio analysé", f"{bench['total_seconds'] / 60:.1f} min")
c[1].metric("Occurrences", bench.get("n_occurrences", "—"))
c[2].metric("Vidéos exclues (vues à l'entraînement)", len(bench.get("forbidden_videos", [])))

rows = []
for model, per_th in bench["results"].items():
    for th, r in per_th.items():
        rows.append({"Modèle": model, "Seuil": th.replace("th", ""),
                     "Rappel": r["recall_stream"], "Détectées": f"{r['detected']}/{r['n_occ']}",
                     "Fausses alarmes": r["false_alarms"], "FA / heure": r["fa_per_hour"],
                     "Incertains": r["uncertain"]})
st.dataframe(pd.DataFrame(rows).style.format({"Rappel": "{:.1%}", "FA / heure": "{:.1f}"}),
             width="stretch", hide_index=True)

st.info("""**Lire les FA/heure honnêtement** : le corpus est thématique (des vidéos
où l'on *parle* d'éloquence), c'est un pire cas volontaire et non une moyenne de
la vie courante. Les événements « incertains » — vérité terrain douteuse — sont
exclus du décompte plutôt que comptés à charge.""")

# ── Audit des erreurs ─────────────────────────────────────────────────────────
st.subheader("Erreurs à écouter")
model = st.selectbox("Modèle", list(bench["results"]))
th = st.selectbox("Seuil", list(bench["results"][model]))
events = bench["results"][model][th].get("events", [])
if not events:
    st.caption("Aucun événement enregistré (banc lancé sans collecte).")
else:
    kinds = st.multiselect("Type", ["FA", "FN", "INCERTAIN"], default=["FA", "FN"])
    filtered = [e for e in events if e["kind"] in kinds]
    st.dataframe(pd.DataFrame(filtered), width="stretch", hide_index=True)
    st.caption(f"{len(filtered)} événement(s). Une erreur de détection se juge à "
               "l'oreille, pas sur un score : ces instants pointent directement les "
               "extraits à réécouter, et les fausses alarmes confirmées deviennent "
               "les *hard negatives* du prochain run.")
