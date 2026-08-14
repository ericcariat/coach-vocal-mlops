"""Page Comparaison — notre détecteur face aux systèmes externes.

Chaque comparaison est un JSON versionné dans `artifacts/reports/comparisons/`
(protocole, grille, réserves, conclusions) : la page les affiche toutes.
Règle du projet : une comparaison n'a de valeur que sur NOTRE banc, avec NOTRE
machine à états et les mêmes règles de comptage — jamais sur les benchmarks
internes des systèmes comparés.

Pour tester un concurrent openWakeWord : déposer sa tête ONNX puis
`coachvocal bench --run <tête>.onnx --minutes 60` (l'adaptateur applique leur
front-end et notre règle de décision), ou l'essayer dans la page Démo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import paths  # noqa: E402

st.set_page_config(page_title="Comparaison", page_icon="⚖️", layout="wide")
st.title("Comparaison avec d'autres systèmes")

comp_dir = paths.REPORTS / "comparisons"
files = sorted(comp_dir.glob("*.json"), reverse=True) if comp_dir.exists() else []

if not files:
    st.info("Aucune comparaison enregistrée.")
    st.code("uv run coachvocal bench --run <tête>.onnx --minutes 60", language="bash")
    st.stop()

st.markdown("""
**Méthode.** Les systèmes externes tournent avec **leur** front-end acoustique
(c'est leur droit d'en avoir un) mais passent par **notre** banc, **notre**
machine à états et **nos** règles de comptage — sinon on compare des
thermomètres, pas des détecteurs. Le rappel par forme (nu / l' / d') fait
partie du verdict.
""")

for f in files:
    data = json.loads(f.read_text())
    st.divider()
    st.subheader(data.get("titre", f.stem))
    st.caption(f"{data.get('date', '')} — {data.get('protocole', '')}")
    if data.get("lignes"):
        df = pd.DataFrame(data["lignes"], columns=data.get("colonnes"))
        st.dataframe(df, width="stretch", hide_index=True)
    if data.get("reserve"):
        st.warning("Réserve méthodologique : " + data["reserve"])
    for c in data.get("conclusions", []):
        st.markdown(f"- {c}")
