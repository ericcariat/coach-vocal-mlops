"""Page Test guidé — mesurer TA voix, essai par essai, dans l'interface.

L'équivalent UI de `coachvocal live guided` : tu enregistres un essai (micro du
navigateur), le modèle choisi le juge HORS LIGNE (mêmes fenêtres, même seuil),
tu confirmes la vérité à l'oreille — waveform + réécoute à l'appui — et le clip
part dans `guided_clips/` nommé TP/FN/FP/TN, le format exact de la source
`guided` (les FN deviennent des positifs durs, les FP des négatifs durs).
Chaque prise passe par la porte qualité (ADR-007) : une prise rejetée (muette,
saturée…) n'est jamais comptée ni sauvegardée.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import paths, registry, runtime  # noqa: E402
from coachvocal.config import QualityGateConfig, load_wakeword  # noqa: E402
from coachvocal.data.gate import judge_clip, measure_clip  # noqa: E402

st.set_page_config(page_title="Test guidé", page_icon="🎯", layout="wide")
st.title("Test guidé — ta voix, essai par essai")

WAKEWORD = "eloquence"
word = load_wakeword(WAKEWORD)
runs = [r["run"] for r in registry.list_runs(WAKEWORD)]
champion = registry.champion_run(WAKEWORD)

c1, c2, c3 = st.columns([2, 1, 2])
run = c1.selectbox("Modèle", runs, index=runs.index(champion) if champion in runs else 0)
threshold = c2.slider("Seuil", 0.1, 0.95, word.live.threshold, 0.05)
attendu = c3.radio("Cet essai est…", ["✅ le mot (« éloquence »)",
                                      "❌ un piège (« éloquente », « élégance »…)"],
                   horizontal=True)
expected_positive = attendu.startswith("✅")

GATE = QualityGateConfig(min_duration_s=0.4, max_duration_s=6.0, tail_check_pools=[])
save_dir = paths.word_dir(WAKEWORD) / "guided_clips"
session_log = paths.word_dir(WAKEWORD) / "guided_clips" / "ui_sessions.json"


@st.cache_resource
def _detector(run_name: str, th: float):
    runtime.configure(use_gpu=False)          # ADR-002
    from coachvocal.inference.detector import load_detector
    return load_detector(paths.run_dir(WAKEWORD, run_name) / "model.keras", word, th)


st.markdown(f"**Consigne :** clique le micro, prononce "
            f"{'« **éloquence** » (nu, naturellement)' if expected_positive else 'le piège choisi'}"
            ", re-clique pour arrêter. Le modèle juge, la waveform s'affiche, TU confirmes.")

st.markdown("""
<style>
div[data-testid="stAudioInput"]{border:3px solid #C0392B;border-radius:14px;
  padding:14px;background:rgba(192,57,43,.06)}
div[data-testid="stAudioInput"] button{transform:scale(1.5);margin:8px 14px}
</style>""", unsafe_allow_html=True)

audio_in = st.audio_input("Essai", key=f"guided_{st.session_state.get('trial_n', 0)}")

if audio_in is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_in.getvalue())
        raw = Path(tmp.name)
    conv = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    conv.close()
    res = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                          "-ar", str(word.sample_rate), "-ac", "1",
                          "-sample_fmt", "s16", conv.name],
                         capture_output=True, text=True)
    raw.unlink(missing_ok=True)
    if res.returncode != 0:
        st.error(f"Conversion échouée : {res.stderr[-200:]}")
        st.stop()

    import soundfile as sf
    audio, sr = sf.read(conv.name, dtype="float32")

    # ── Porte qualité d'abord : une prise pourrie ne compte pas ──────────────
    m = measure_clip(Path(conv.name), word.sample_rate)
    verdict_q, raisons = judge_clip(m, GATE, word.sample_rate, pool="guided")
    if verdict_q == "rejete":
        st.error("Prise rejetée par la porte qualité : " + " · ".join(raisons)
                 + " — refais l'essai.")
        Path(conv.name).unlink(missing_ok=True)
        st.stop()

    # ── Verdict du modèle (hors ligne, mêmes fenêtres que le live) ───────────
    detector = _detector(run, threshold)
    probas, peaks, _ = detector.window_probas(audio)
    p_max = float(probas.max()) if len(probas) else 0.0
    triggers = detector.triggers_from(probas, peaks, threshold)
    detected = bool(triggers) or p_max > threshold

    # ── Waveform + réécoute ──────────────────────────────────────────────────
    env = np.abs(audio)
    step = max(1, len(env) // 900)
    env = env[: len(env) // step * step].reshape(-1, step).max(axis=1)
    st.area_chart({"amplitude": env.tolist()}, height=140)
    wcol, vcol = st.columns([3, 2])
    wcol.audio(conv.name)
    vcol.metric("Verdict du modèle",
                "🚨 DÉTECTÉ" if detected else "— pas détecté",
                f"proba max {p_max:.0%}")
    if verdict_q == "douteux":
        st.warning("Qualité limite : " + " · ".join(raisons) + " — à toi de juger.")
    st.caption(f"Durée {m['duration_s']} s · RMS {m['rms']} · pic {m['peak']} · "
               f"SNR {m['snr_db']} dB · saturation {m['saturation_ratio']:.1%}")

    outcome = {(True, True): "TP", (True, False): "FN",
               (False, True): "FP", (False, False): "TN"}[(expected_positive, detected)]
    LIBELLE = {"TP": "✅ TP — le mot, détecté", "FN": "🔇 FN — le mot, RATÉ",
               "FP": "🚨 FP — un piège, détecté à tort", "TN": "✅ TN — un piège, ignoré"}
    st.subheader(LIBELLE[outcome])

    ok_col, redo_col = st.columns(2)
    if ok_col.button(f"💾 Valider et enregistrer ({outcome})", type="primary"):
        save_dir.mkdir(parents=True, exist_ok=True)
        name = f"guided_{dt.datetime.now():%H%M%S_%f}_{outcome}.wav"
        sf.write(save_dir / name, audio, word.sample_rate, subtype="PCM_16")
        log = json.loads(session_log.read_text()) if session_log.exists() else []
        log.append({"date": dt.datetime.now().isoformat(timespec="seconds"),
                    "clip": name, "outcome": outcome, "model": run,
                    "threshold": threshold, "p_max": round(p_max, 4),
                    "gate": verdict_q, "mesures": m})
        session_log.write_text(json.dumps(log, indent=1, ensure_ascii=False))
        Path(conv.name).unlink(missing_ok=True)
        st.session_state["trial_n"] = st.session_state.get("trial_n", 0) + 1
        st.rerun()
    if redo_col.button("🔁 Refaire (ne pas compter)"):
        Path(conv.name).unlink(missing_ok=True)
        st.session_state["trial_n"] = st.session_state.get("trial_n", 0) + 1
        st.rerun()

# ── Score de la session ───────────────────────────────────────────────────────
if session_log.exists():
    log = json.loads(session_log.read_text())
    today = dt.date.today().isoformat()
    mine = [t for t in log if t["date"].startswith(today) and t["model"] == run]
    if mine:
        tp = sum(1 for t in mine if t["outcome"] == "TP")
        fn = sum(1 for t in mine if t["outcome"] == "FN")
        fp = sum(1 for t in mine if t["outcome"] == "FP")
        tn = sum(1 for t in mine if t["outcome"] == "TN")
        st.divider()
        st.subheader(f"Session du jour — {run} @ {threshold}")
        s1, s2, s3 = st.columns(3)
        s1.metric("TA voix : rappel", f"{tp}/{tp + fn}" if tp + fn else "—",
                  f"{tp / (tp + fn):.0%}" if tp + fn else None)
        s2.metric("Pièges ignorés", f"{tn}/{fp + tn}" if fp + tn else "—")
        s3.metric("Essais", len(mine))
        with st.expander("Réécouter les essais du jour"):
            for t in reversed(mine):
                left, right = st.columns([2, 4])
                left.markdown(f"`{t['outcome']}` — proba {t['p_max']:.0%}")
                wav = save_dir / t["clip"]
                if wav.exists():
                    right.audio(str(wav))
