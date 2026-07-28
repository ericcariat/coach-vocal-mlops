"""Page Données — composition du jeu, provenance, audit qualité."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import paths  # noqa: E402
from coachvocal.config import list_experiments, load_experiment  # noqa: E402
from coachvocal.data import sources as src_registry  # noqa: E402

st.set_page_config(page_title="Données", page_icon="📊", layout="wide")
st.title("📊 Données")

experiment = st.selectbox("Expérience", list_experiments())
cfg = load_experiment(experiment)

st.caption(f"Recette **{cfg.dataset.name}** · seed données `{cfg.dataset.data_seed}` "
           "(jamais modifiée : elle fige le jeu de test)")

# ── Recette ───────────────────────────────────────────────────────────────────
st.subheader("Recette du dataset")
rows = [{"Source": s.name, "Type": s.type, "Label": "positif" if s.label else "négatif",
         "Boost (×copies)": s.copies, "Splits": ", ".join(s.splits),
         "Active": "✅" if s.enabled else "—",
         "Paramètres": json.dumps(s.params, ensure_ascii=False)}
        for s in cfg.dataset.sources]
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with st.expander("Comment lire le « boost »"):
    st.markdown("""
Le boost duplique les chemins d'une source rare (ma voix, les négatifs proches)
pour la faire peser davantage, **sans toucher à la fonction de coût**. C'est plus
lisible qu'un poids par échantillon et ça se relit dans le manifest.

Le **class_weight**, lui, corrige le déséquilibre global positifs/négatifs
(1 pour 20 à 40). Sans lui, prédire toujours « non » donnerait 97 % d'accuracy —
c'est pourquoi l'accuracy n'est pas un critère ici.
""")

st.subheader("Augmentation (train uniquement)")
c1, c2 = st.columns(2)
c1.metric("Décalage temporel", f"±{cfg.dataset.augmentation.time_shift_ms} ms")
c2.metric("Vitesse d'élocution",
          f"×{cfg.dataset.augmentation.speed_min} → ×{cfg.dataset.augmentation.speed_max}")
st.caption("La variation de vitesse vient d'un diagnostic : 15 « éloquence » sur 20 "
           "prononcés au tempo réel étaient ratés, mais 20/20 reconnus une fois "
           "ralentis de 8-15 %. Le modèle avait appris un débit, pas un mot.")

# ── Construction ──────────────────────────────────────────────────────────────
st.subheader("Construire et auditer")
st.code(f"uv run coachvocal data build {experiment}\n"
        f"uv run coachvocal data audit {experiment}", language="bash")

audit_json = paths.report_dir("data_quality") / f"{cfg.dataset.name}.json"
if audit_json.exists():
    rep = json.loads(audit_json.read_text())
    st.success("✅ Aucune anomalie détectée") if rep["ok"] else st.warning(
        "⚠️ " + " · ".join(rep["issues"][:6]))
    png = audit_json.with_suffix(".png")
    if png.exists():
        st.image(str(png))
    st.dataframe(pd.DataFrame([
        {"Pool": k, "Fichiers": v["n_files"], "Pic médian": round(v["peak_median"], 3),
         "Durée médiane (s)": round(v["duration_median"], 2),
         "Problèmes": ", ".join(f"{n}× {kind}" for kind, n in v["problems"].items()) or "—"}
        for k, v in rep["pools"].items()]), width="stretch", hide_index=True)
else:
    st.info("Aucun audit enregistré pour cette recette.")

# ── Sources disponibles ───────────────────────────────────────────────────────
with st.expander("Types de sources disponibles"):
    st.write(", ".join(f"`{s}`" for s in src_registry.available()))
    st.caption("Ajouter une source = un module décoré `@source(\"nom\")`. "
               "Aucune modification du pipeline n'est nécessaire.")
