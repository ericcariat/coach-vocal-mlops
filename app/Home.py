"""Interface Streamlit — vue d'ensemble du projet.

Objectif : rendre le pipeline manipulable et surtout LISIBLE. Les
hyperparamètres sont affichés et modifiables, pas cachés dans le code : montrer
qu'on sait quel réglage agit sur quoi fait partie du livrable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coachvocal import registry  # noqa: E402
from coachvocal.config import list_experiments, load_experiment  # noqa: E402

st.set_page_config(page_title="Coach vocal — wake word", page_icon="🎙", layout="wide")

WAKEWORD = st.sidebar.selectbox("Mot-clé", ["eloquence"])
st.sidebar.caption("Pages : Données · Entraînement · Évaluation · Banc streaming · Démo")

st.title("🎙 Coach vocal — détection de wake word")
st.caption("Première brique d'un coach vocal : micro → CNN binaire → déclenchement.")

# ── Champion ──────────────────────────────────────────────────────────────────
reg = registry.load(WAKEWORD)
champion = reg.get("champion")
runs = registry.list_runs(WAKEWORD)
bench = registry.bench_results(WAKEWORD)

if champion:
    st.success(f"**Modèle en production : `{champion['run']}`** "
               f"(promu le {champion['promoted']})\n\n{champion['reason']}")
else:
    st.warning("Aucun champion promu. `coachvocal train <experience>` puis "
               "`coachvocal registry promote <run> --reason \"…\"`.")

cols = st.columns(4)
if champion and bench.get("results", {}).get(champion["run"]):
    r = bench["results"][champion["run"]].get("th0.8") or \
        next(iter(bench["results"][champion["run"]].values()))
    cols[0].metric("Rappel streaming", f"{r['recall_stream']:.1%}")
    cols[1].metric("Fausses alarmes / h", f"{r['fa_per_hour']:.1f}")
    cols[2].metric("Occurrences testées", r["n_occ"])
    cols[3].metric("Audio continu", f"{bench['total_seconds'] / 60:.0f} min")
else:
    cols[0].metric("Runs enregistrés", len(runs))
    cols[1].metric("Expériences", len(list_experiments()))
    cols[2].metric("Banc streaming", "non lancé")
    cols[3].metric("Champion", champion["run"] if champion else "—")

# ── Pipeline ──────────────────────────────────────────────────────────────────
st.subheader("Le pipeline, étape par étape")
st.markdown("""
| # | Étape | Ce qui s'y joue | Commande |
|---|---|---|---|
| 1 | **Collecte** | clips réels (YouTube, ma voix), corpus externes, positifs TTS | `coachvocal data tts-pool` |
| 2 | **Split figé** | séparation *par groupe* (vidéo/session), écrite une seule fois | `coachvocal data split` |
| 3 | **Recette** | sources + pondérations décrites en YAML → manifest | `coachvocal data build` |
| 4 | **Audit qualité** | durées, niveaux, doublons, fuites — avant d'entraîner | `coachvocal data audit` |
| 5 | **Entraînement** | N candidats, élection **par la validation** | `coachvocal train` |
| 6 | **Évaluation clip** | accuracy, F1, FRR, FAR, AUC — signal de contrôle | *(inclus)* |
| 7 | **Banc streaming** | conditions réelles — **c'est lui qui décide** | `coachvocal bench` |
| 8 | **Promotion** | champion tracé, `current/` mis à jour | `coachvocal registry promote` |
| 9 | **Service** | API FastAPI + micro always-on | `coachvocal serve` · `coachvocal live listen` |
""")

st.info("""**Pourquoi deux évaluations ?** Le test par clips a déjà **mal classé** les
modèles : le meilleur en F1 par clip s'est révélé le pire en conditions réelles.
Un clip d'1 s parfaitement centré n'existe pas en production — en production il y
a des demi-mots, du recouvrement et une décision à prendre 8 fois par seconde.""")

# ── Expériences ───────────────────────────────────────────────────────────────
st.subheader("Expériences déclarées")
for name in list_experiments():
    cfg = load_experiment(name)
    with st.expander(f"**{name}** — {cfg.description or 'sans description'}"):
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Dataset** `{cfg.dataset.name}`\n\n{len(cfg.dataset.sources)} sources")
        c2.write(f"**Modèle** `{cfg.model.arch}`\n\n{cfg.model.params}")
        c3.write(f"**Entraînement**\n\n{cfg.training.epochs} epochs · "
                 f"batch {cfg.training.batch_size} · seeds {cfg.training.seeds}")
        st.code(f"uv run coachvocal train {name}", language="bash")
