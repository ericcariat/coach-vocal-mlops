"""Page Entraînement — hyperparamètres explicites, lancement, journal en direct.

Les hyperparamètres sont exposés délibérément : savoir lequel agit sur quoi, et
pouvoir le montrer, fait partie de ce qu'on doit démontrer.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coachvocal.config import list_experiments, load_experiment  # noqa: E402

st.set_page_config(page_title="Entraînement", page_icon="🚀", layout="wide")
st.title("Entraînement")

experiment = st.selectbox("Expérience", list_experiments())
cfg = load_experiment(experiment)

st.subheader("Hyperparamètres")
c1, c2, c3 = st.columns(3)
epochs = c1.number_input("Epochs (max)", 1, 200, cfg.training.epochs)
batch = c2.selectbox("Batch size", [16, 32, 64, 128],
                     index=[16, 32, 64, 128].index(cfg.training.batch_size))
lr = c3.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                      value=cfg.training.learning_rate)
c4, c5, c6 = st.columns(3)
patience = c4.number_input("Patience EarlyStopping", 1, 20, cfg.training.early_stopping_patience)
n_seeds = c5.number_input("Nombre de candidats (seeds)", 1, 10, len(cfg.training.seeds))
threshold = c6.slider("Seuil d'évaluation", 0.1, 0.9, cfg.training.threshold, 0.05)

with st.expander("À quoi sert chaque réglage", expanded=False):
    st.markdown("""
- **Epochs / patience** — l'entraînement s'arrête lorsqu'il n'y a plus
  d'amélioration et conserve la meilleure version du modèle.
- **Batch size** — au-delà de 128 sur CPU, chaque epoch s'allonge sans gain de qualité.
- **Learning rate** — 1e-3 (Adam) est le point stable ; à 1e-2 la loss diverge.
- **Nombre de candidats** — parade à la non-reproductibilité CPU (±0.03-0.06 de F1
  à seed identique). On entraîne N modèles et on élit **par la validation**.
- **Seuil** — 0.5 pour l'évaluation, 0.8 en production : en always-on, une fausse
  alarme coûte plus cher qu'un mot à répéter.
""")

st.warning("⚠️ Entraînement **sur CPU** : le plugin tensorflow-metal corrompt les "
           "gradients sur cette machine (la loss explose). Voir `docs/decisions/ADR-002`.")

seeds = cfg.training.seeds[:n_seeds] if n_seeds <= len(cfg.training.seeds) else \
    list(range(42, 42 + int(n_seeds)))
run_id = st.text_input("Identifiant du run", experiment)

cmd = ["uv", "run", "coachvocal", "train", experiment, "--run-id", run_id,
       "--set", f"training.epochs={epochs}", "--set", f"training.batch_size={batch}",
       "--set", f"training.learning_rate={lr}", "--set", f"training.threshold={threshold}",
       "--set", f"training.early_stopping_patience={patience}",
       "--set", f"training.seeds={seeds}"]
st.code(" ".join(cmd), language="bash")
st.caption("La commande est reproductible telle quelle en dehors de l'interface.")

if st.button("▶️ Lancer l'entraînement", type="primary"):
    log = st.empty()
    lines: list[str] = []
    with st.spinner(f"{len(seeds)} candidat(s) — plusieurs minutes…"):
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            lines.append(line.rstrip())
            log.code("\n".join(lines[-30:]))
        proc.wait()
    if proc.returncode == 0:
        st.success("Terminé — voir la page **Évaluation**.")
    else:
        st.error(f"Échec (code {proc.returncode}).")
