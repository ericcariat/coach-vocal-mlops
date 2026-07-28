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
st.title("📈 Évaluation")

WAKEWORD = "eloquence"
runs = registry.list_runs(WAKEWORD)
if not runs:
    st.info("Aucun run. Lancer un entraînement depuis la page précédente.")
    st.stop()

st.subheader("Comparatif — toutes les métriques")
df = pd.DataFrame([{
    "Run": r["run"] + (" ⭐" if r["is_champion"] else ""),
    "Date": (r["date"] or "")[:10], "Seed": r["seed"],
    "Accuracy": r["accuracy"], "F1": r["f1"], "FRR ↓": r["frr"], "FAR ↓": r["far"],
    "ROC-AUC": r["roc_auc"], "Empreinte data": r["dataset_fingerprint"],
} for r in runs])
st.dataframe(df.style.format({"Accuracy": "{:.2%}", "F1": "{:.4f}", "FRR ↓": "{:.2%}",
                              "FAR ↓": "{:.2%}", "ROC-AUC": "{:.4f}"}, na_rep="—"),
             width="stretch", hide_index=True)

st.caption("**FRR** : le mot est prononcé, rien ne se passe. **FAR** : un négatif "
           "déclenche. Deux runs de même *empreinte data* ont vu exactement les mêmes "
           "fichiers : un écart de métriques sans écart d'empreinte est de la variance.")

st.error("⚠️ Ces chiffres décrivent des clips d'1 s pré-découpés. Ils ont déjà **mal "
         "classé** les modèles — la décision de promotion se prend au **banc streaming**.")

# ── Détail d'un run ───────────────────────────────────────────────────────────
run = st.selectbox("Détail du run", [r["run"] for r in runs])
run_dir = paths.run_dir(WAKEWORD, run)
metrics = json.loads((run_dir / "metrics.json").read_text())

c = st.columns(5)
test = metrics.get("test", {})
c[0].metric("F1", f"{test.get('f1_pos', 0):.4f}")
c[1].metric("FRR", f"{test.get('frr', 0):.2%}")
c[2].metric("FAR", f"{test.get('far', 0):.2%}")
c[3].metric("ROC-AUC", f"{test.get('roc_auc', 0):.4f}")
c[4].metric("Seed élue", metrics.get("selected_seed", "—"))

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
