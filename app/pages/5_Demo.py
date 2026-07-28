"""Page Démo — tester le champion sur un fichier audio, via l'API ou en direct."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import paths, registry, runtime  # noqa: E402
from coachvocal.config import load_wakeword  # noqa: E402

st.set_page_config(page_title="Démo", page_icon="🎤", layout="wide")
st.title("🎤 Démo")

WAKEWORD = "eloquence"
word = load_wakeword(WAKEWORD)
runs = [r["run"] for r in registry.list_runs(WAKEWORD)]
if not runs:
    st.info("Aucun modèle entraîné.")
    st.stop()

champion = registry.champion_run(WAKEWORD)
run = st.selectbox("Modèle", runs, index=runs.index(champion) if champion in runs else 0)
threshold = st.slider("Seuil de décision", 0.1, 0.99, word.live.threshold, 0.01)

uploaded = st.file_uploader("Fichier audio (WAV mono)", type=["wav"])
if uploaded:
    st.audio(uploaded)
    if st.button("Analyser", type="primary"):
        import io

        import soundfile as sf

        runtime.configure(use_gpu=False)     # Metal fausse les probas (ADR-002)
        from coachvocal.inference.detector import load_detector

        with st.spinner("Chargement du modèle…"):
            detector = load_detector(paths.run_dir(WAKEWORD, run) / "model.keras",
                                     word, threshold)
        audio, sr = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != word.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=word.sample_rate)

        probas, peaks, starts = detector.window_probas(audio)
        triggers = detector.triggers_from(probas, peaks, threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité max", f"{probas.max():.1%}")
        c2.metric("Déclenchements", len(triggers))
        c3.metric("Durée", f"{len(audio) / word.sample_rate:.1f} s")

        times = starts / word.sample_rate + word.audio.clip_seconds
        st.line_chart({"probabilité": probas.tolist(),
                       "seuil": [threshold] * len(probas)})
        st.caption(f"Une décision toutes les {word.live.hop_s * 1000:.0f} ms. "
                   f"Il faut {word.live.n_consecutive} fenêtres consécutives au-dessus "
                   "du seuil pour déclencher — un pic isolé ne suffit pas.")
        if triggers:
            st.success("Déclenchements à : " + ", ".join(f"{t:.2f} s" for t in triggers))
        else:
            st.warning("Aucun déclenchement.")
        if float(probas.max()) > threshold and not triggers:
            st.info("La probabilité dépasse le seuil mais la règle des fenêtres "
                    "consécutives (ou le portail d'énergie) a filtré l'événement — "
                    "exactement ce qui évite la plupart des fausses alarmes.")
        _ = times

st.divider()
st.subheader("En ligne de commande")
st.code("uv run coachvocal live listen                # micro always-on\n"
        "uv run coachvocal live guided                # test guidé essai par essai\n"
        "uv run coachvocal serve                      # API + Swagger sur /docs",
        language="bash")
