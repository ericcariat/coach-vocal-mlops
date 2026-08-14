"""Page Démo — tester le champion sur un fichier audio, via l'API ou en direct."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import registry, runtime  # noqa: E402
from coachvocal.config import load_wakeword  # noqa: E402

st.set_page_config(page_title="Démo", page_icon="🎤", layout="wide")
st.title("Démo")

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
# Candidats hors registre : « v28_resserre_seed46 (oww) » — la version d'abord,
# pour que la liste se classe naturellement avec les runs du registre. Un
# candidat déjà intégré comme run n'apparaît qu'une fois (sous son run).
oww_par_label = {f"{p.stem} (oww)": p for p in oww_heads
                 if not any(p.stem.startswith(r) for r in runs)}
options = sorted(runs + list(oww_par_label))
run = st.selectbox("Modèle", options,
                   index=options.index(champion) if champion in options else 0,
                   format_func=lambda o: f"{o} ⭐" if o == champion else o)
threshold = st.slider("Seuil de décision", 0.01, 0.99, word.live.threshold, 0.01)


def _load_selected_detector():
    runtime.configure(use_gpu=False)         # Metal fausse les probas (ADR-002)
    if run in oww_par_label:
        from coachvocal.evaluation.oww_adapter import OwwDetector
        return OwwDetector(oww_par_label[run], word, threshold)
    from coachvocal.inference.detector import load_detector
    return load_detector(registry.model_path(WAKEWORD, run), word, threshold)

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
st.subheader("Écoute en direct")
st.caption("Le micro de CETTE machine (Streamlit tourne en local). Même modèle, "
           "même seuil et même machine à états que le banc et `live listen`.")

try:
    from coachvocal.inference.live import list_devices
    devices = list_devices()
except Exception:
    devices = []

# Les micros « Continuité » (iPhone/iPad) déclenchent un appairage Bluetooth
# lent dès qu'on les ouvre : on les écarte de la liste. Défaut : le micro
# intégré du MacBook s'il existe, sinon l'entrée par défaut du système.
devices = [d for d in devices if "iphone" not in d["name"].lower()
           and "ipad" not in d["name"].lower()]

if not devices:
    st.warning("Aucun micro détecté (ou `sounddevice` indisponible).")
else:
    default_idx = next((i for i, d in enumerate(devices)
                        if "macbook" in d["name"].lower()), None)
    if default_idx is None:
        try:
            import sounddevice as _sd
            sys_default = _sd.default.device[0]
            default_idx = next((i for i, d in enumerate(devices)
                                if d["index"] == sys_default), 0)
        except Exception:
            default_idx = 0
    dev = st.selectbox("Micro", devices, index=default_idx,
                       format_func=lambda d: f"{d['index']} — {d['name']}")

    # La capture tourne dans un THREAD (l'interface reste réactive) ; un
    # fragment Streamlit se rafraîchit tout seul pour afficher l'état.
    class _Listener:
        def __init__(self, detector, sample_rate: int, device: int):
            self.detector = detector
            self.sample_rate = sample_rate
            self.device = device
            self.running = False
            self.proba = 0.0
            self.peak = 0.0
            self.detections: list[str] = []
            self.last_trigger = 0.0
            self.error: str | None = None

        def start(self):
            import threading
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()

        def stop(self):
            self.running = False

        def _loop(self):
            import queue as _queue
            import time

            import sounddevice as sd
            q: _queue.Queue = _queue.Queue()

            def _cb(indata, frames, time_info, s):  # noqa: ARG001
                q.put(indata[:, 0].copy())

            try:
                with sd.InputStream(samplerate=self.sample_rate,
                                    blocksize=self.detector.hop, channels=1,
                                    dtype="float32", device=self.device,
                                    callback=_cb):
                    while self.running:
                        try:
                            chunk = q.get(timeout=0.5)
                        except _queue.Empty:
                            continue
                        event = self.detector.push(chunk)
                        if not event:
                            continue
                        self.proba, self.peak = float(event["proba"]), float(event["peak"])
                        if event["triggered"]:
                            self.last_trigger = time.monotonic()
                            self.detections.append(time.strftime("%H:%M:%S"))
            except Exception as exc:                     # micro débranché, etc.
                self.error = str(exc)
                self.running = False

    listener = st.session_state.get("demo_listener")
    en_cours = listener is not None and listener.running

    c_start, c_stop = st.columns(2)
    if c_start.button("▶️ Démarrer l'écoute", type="primary", disabled=en_cours):
        with st.spinner("Chargement du modèle…"):
            detector = _load_selected_detector()
        listener = _Listener(detector, word.sample_rate, dev["index"])
        st.session_state["demo_listener"] = listener
        listener.start()
        st.rerun()
    if c_stop.button("⏹ Arrêter", disabled=not en_cours):
        listener.stop()
        st.rerun()

    @st.fragment(run_every=0.4 if en_cours else None)
    def _live_status():
        import time
        lst = st.session_state.get("demo_listener")
        if lst is None:
            st.caption("Prêt — choisis un modèle et démarre l'écoute.")
            return
        if lst.error:
            st.error(f"Capture interrompue : {lst.error}")
            return
        if lst.running and time.monotonic() - lst.last_trigger < 2.5:
            st.success(f"## 🚨 « {WAKEWORD} » détecté !")
        elif lst.running:
            st.info(f"🎙 J'écoute… — {len(lst.detections)} détection(s)")
        else:
            st.info(f"🏁 Session terminée — {len(lst.detections)} détection(s).")
        st.progress(min(1.0, lst.proba),
                    text=f"probabilité {lst.proba:.0%} · pic {lst.peak:.2f}")
        if lst.detections:
            st.success("Détections à : " + " · ".join(lst.detections))

    _live_status()

st.divider()
st.subheader("En ligne de commande")
st.code("uv run coachvocal live listen                # micro always-on\n"
        "uv run coachvocal live guided                # test guidé essai par essai\n"
        "uv run coachvocal serve                      # API + Swagger sur /docs",
        language="bash")
