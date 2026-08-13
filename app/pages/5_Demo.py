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
# Concurrents openWakeWord (têtes ONNX) : testables dans la même démo, via
# l'adaptateur du banc — leur front-end, notre machine à états.
OWW_COMPARE = Path(__file__).resolve().parents[2] / "open_wake_word_compare"
oww_heads = sorted(OWW_COMPARE.glob("*.onnx")) if OWW_COMPARE.exists() else []
options = runs + [f"oww : {p.name}" for p in oww_heads]
run = st.selectbox("Modèle", options, index=runs.index(champion) if champion in runs else 0)
threshold = st.slider("Seuil de décision", 0.01, 0.99, word.live.threshold, 0.01)
if run.startswith("oww : ") and threshold > 0.35:
    st.info("💡 Les têtes openWakeWord sont calibrées BAS : au banc, leur rappel "
            "plafonne déjà à 0.5 et s'améliore vers 0.05-0.3. Baisse le seuil "
            "pour leur laisser une chance.")


def _load_selected_detector():
    runtime.configure(use_gpu=False)         # Metal fausse les probas (ADR-002)
    if run.startswith("oww : "):
        from coachvocal.evaluation.oww_adapter import OwwDetector
        return OwwDetector(OWW_COMPARE / run[len("oww : "):], word, threshold)
    from coachvocal.inference.detector import load_detector
    return load_detector(paths.run_dir(WAKEWORD, run) / "model.keras", word, threshold)

uploaded = st.file_uploader("Fichier audio (WAV mono)", type=["wav"])
if uploaded:
    st.audio(uploaded)
    if st.button("Analyser", type="primary"):
        import io

        import soundfile as sf

        with st.spinner("Chargement du modèle…"):
            detector = _load_selected_detector()
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

# ── Écoute en direct au micro ─────────────────────────────────────────────────
st.divider()
st.subheader("🎙 Écoute en direct")
st.caption("Le micro de CETTE machine (Streamlit tourne en local). Même modèle, "
           "même seuil et même machine à états que le banc et `live listen`.")

try:
    from coachvocal.inference.live import list_devices
    devices = list_devices()
except Exception:
    devices = []

if not devices:
    st.warning("Aucun micro détecté (ou `sounddevice` indisponible).")
else:
    dcol, tcol = st.columns([3, 1])
    dev = dcol.selectbox("Micro", devices,
                         format_func=lambda d: f"{d['index']} — {d['name']}")
    duration = tcol.number_input("Durée (s)", 10, 180, 30, 5)

    if st.button("▶️ Démarrer l'écoute", type="primary"):
        import queue as _queue
        import time

        import sounddevice as sd

        with st.spinner("Chargement du modèle…"):
            detector = _load_selected_detector()
        status = st.empty()
        gauge = st.empty()
        journal = st.empty()

        q: _queue.Queue = _queue.Queue()

        def _cb(indata, frames, time_info, s):  # noqa: ARG001
            q.put(indata[:, 0].copy())

        detections: list[str] = []
        flash_until = 0.0
        t_end = time.monotonic() + duration
        with sd.InputStream(samplerate=word.sample_rate, blocksize=detector.hop,
                            channels=1, dtype="float32", device=dev["index"],
                            callback=_cb):
            while time.monotonic() < t_end:
                try:
                    chunk = q.get(timeout=0.5)
                except _queue.Empty:
                    continue
                event = detector.push(chunk)
                now = time.monotonic()
                if event and event["triggered"]:
                    flash_until = now + 2.5      # le bandeau reste ~2,5 s
                    detections.append(time.strftime("%H:%M:%S"))
                if flash_until > now:
                    status.success(f"## 🚨 « {WAKEWORD} » détecté !")
                else:
                    reste = int(t_end - now)
                    status.info(f"🎙 J'écoute… ({reste} s restantes) — "
                                f"{len(detections)} détection(s)")
                if event:
                    gauge.progress(min(1.0, float(event["proba"])),
                                   text=f"probabilité {event['proba']:.0%} · "
                                        f"pic {event['peak']:.2f}")
        status.info(f"🏁 Session terminée — {len(detections)} détection(s).")
        if detections:
            journal.success("Détections à : " + " · ".join(detections))

st.divider()
st.subheader("En ligne de commande")
st.code("uv run coachvocal live listen                # micro always-on\n"
        "uv run coachvocal live guided                # test guidé essai par essai\n"
        "uv run coachvocal serve                      # API + Swagger sur /docs",
        language="bash")
