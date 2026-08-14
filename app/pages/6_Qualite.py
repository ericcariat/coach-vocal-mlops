"""Page Qualité — la file d'audit humain de la porte qualité (ADR-007).

La machine a trié : les acceptés sont entrés, les rejetés sont listés ici pour
contrôle, et les DOUTEUX attendent un verdict humain. Un « oui » réintègre le
clip au prochain build, un « non » l'exclut définitivement. Les verdicts sont
persistés dans `gate_human.json`, à côté du rapport de la porte.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal.data.gate import HUMAN_NAME, REPORT_NAME, gate_dir  # noqa: E402

st.set_page_config(page_title="Qualité", page_icon="🚪", layout="wide")
st.title("Porte qualité")

WAKEWORD = "eloquence"
report_path = gate_dir(WAKEWORD) / REPORT_NAME
human_path = gate_dir(WAKEWORD) / HUMAN_NAME

if not report_path.exists():
    st.info("Aucune porte n'a encore tourné.")
    st.code("uv run coachvocal data gate <experiment>", language="bash")
    st.stop()

report = json.loads(report_path.read_text())
human: dict = json.loads(human_path.read_text()) if human_path.exists() else {}

# ── Sauvegarde AVANT affichage (état des radios en session_state) ─────────────
OPTIONS = ["— à juger —", "✅ Oui, garder", "❌ Non, exclure"]
VERDICT_CODE = {"✅ Oui, garder": "oui", "❌ Non, exclure": "non"}
changed = False
for path in report["clips"]:
    choice = st.session_state.get(f"gate_{path}")
    if choice is None:
        continue
    saved = human.get(path, {}).get("verdict")
    code = VERDICT_CODE.get(choice)
    if code != saved:
        if code is None:
            human.pop(path, None)
        else:
            human[path] = {"verdict": code,
                           "saved_at": datetime.date.today().isoformat()}
        changed = True
if changed:
    human_path.write_text(json.dumps(human, indent=1, ensure_ascii=False))

c = report["counts"]
cols = st.columns(4)
cols[0].metric("Acceptés", c["accepte"])
cols[1].metric("Rejetés (auto)", c["rejete"])
cols[2].metric("Douteux", c["douteux"])
cols[3].metric("Douteux tranchés", sum(1 for v in human.values() if v.get("verdict")))

st.markdown("""
- **Accepté** : dans les seuils, entre dans le jeu de données sans intervention.
- **Rejeté** : hors seuils francs (muet, saturé, mauvaise durée…), écarté
  automatiquement — contrôlable ci-dessous, jamais supprimé du disque.
- **Douteux** : zone grise — c'est TON verdict qui décide. Tant qu'un douteux
  n'est pas tranché, il est exclu du build (politique `doubt_policy: exclude`).
""")

# ── File des douteux ──────────────────────────────────────────────────────────
st.subheader("Douteux — à trancher")
doubtful = {p: cl for p, cl in report["clips"].items() if cl["verdict"] == "douteux"}
if not doubtful:
    st.success("Aucun douteux — la machine a tout tranché.")
else:
    only_open = st.checkbox("Masquer les douteux déjà tranchés", value=True)
    shown = 0
    for path, cl in sorted(doubtful.items()):
        h = human.get(path, {})
        if only_open and h.get("verdict"):
            continue
        shown += 1
        left, mid, right = st.columns([3, 3, 3], vertical_alignment="center")
        with left:
            st.markdown(f"`{Path(path).name}`")
            st.caption(f"{cl['pool']} — " + " · ".join(cl["raisons"]))
        with mid:
            if Path(path).exists():
                st.audio(path)
            else:
                st.warning("fichier introuvable")
        with right:
            idx = 0
            if h.get("verdict") == "oui":
                idx = 1
            elif h.get("verdict") == "non":
                idx = 2
            st.radio("Verdict", OPTIONS, index=idx, key=f"gate_{path}",
                     horizontal=True, label_visibility="collapsed")
            if h.get("saved_at"):
                st.caption(f"Enregistré le {h['saved_at']}")
        st.divider()
    if shown == 0:
        st.success("Tous les douteux sont tranchés.")

# ── Rejetés (contrôle) ────────────────────────────────────────────────────────
with st.expander(f"🗑️ Rejetés automatiquement ({c['rejete']}) — contrôle"):
    rows = [{"Fichier": Path(p).name, "Pool": cl["pool"],
             "Raisons": " · ".join(cl["raisons"])}
            for p, cl in sorted(report["clips"].items()) if cl["verdict"] == "rejete"]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Un rejet automatique se conteste en ajustant les seuils dans "
                   "la config (`dataset.quality_gate`) et en relançant la porte — "
                   "pas en éditant ce fichier.")
    else:
        st.caption("Aucun rejet.")
