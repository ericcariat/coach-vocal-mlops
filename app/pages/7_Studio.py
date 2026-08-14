"""Page Studio — campagne d'enregistrement guidée du mot-clé (ROADMAP P2).

Inspirée de la console ViolaWake : des prises scriptées par condition (voix,
débit, distance), un contrôle qualité IMMÉDIAT à la prise (les mêmes mesures
que la porte qualité, ADR-007), et une règle d'or : **une session = un groupe
indivisible**, qui n'alimente que le train (source `studio`).

Chaque prise validée est convertie en 16 kHz mono PCM16 et consignée dans
`data/wakewords/<mot>/studio/<session>/` avec `metadata.json` (mesures,
verdict, horodatage). La source `studio` n'utilise QUE les prises `keep: true`.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import paths  # noqa: E402
from coachvocal.config import QualityGateConfig, load_wakeword  # noqa: E402
from coachvocal.data.gate import judge_clip, measure_clip  # noqa: E402

st.set_page_config(page_title="Studio", page_icon="🎙️", layout="wide")
st.title("Studio d'enregistrement")

WAKEWORD = "eloquence"
word = load_wakeword(WAKEWORD)

# Conditions de la campagne — l'ordre est le script de la session.
CAMPAGNE = [
    ("normal", "Voix normale, distance habituelle de l'ordinateur"),
    ("fort", "Voix forte, comme pour être entendu d'une autre pièce"),
    ("joyeux", "Ton joyeux, enthousiaste"),
    ("rapide", "Débit rapide, comme pressé"),
    ("lent", "Débit lent, en détachant les syllabes"),
    ("50cm", "À environ 50 cm du micro"),
    ("1m", "À environ 1 mètre du micro"),
    ("2m50", "À environ 2,5 mètres du micro (voix portée)"),
]

# Prises plus longues qu'un clip (le recadrage 1 s vient de la source `studio`),
# et pas de contrôle de fin chargée : on parle librement autour du mot.
GATE = QualityGateConfig(min_duration_s=0.5, max_duration_s=6.0, tail_check_pools=[])

studio_root = paths.word_dir(WAKEWORD) / "studio"

with st.sidebar:
    st.subheader("Session")
    default_session = datetime.date.today().isoformat()
    session_name = st.text_input("Nom de session (= groupe de split)", value=default_session)
    takes_per_cond = st.number_input("Prises par condition", 1, 30, 10)
    note = st.text_input("Matériel / contexte (micro, pièce…)", value="")
    st.caption("Une session est un groupe INDIVISIBLE : toutes ses prises vont "
               "au train, jamais en val/test.")

session_dir = studio_root / session_name
meta_path = session_dir / "metadata.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"takes": {}}


def save_meta():
    session_dir.mkdir(parents=True, exist_ok=True)
    meta["session"] = session_name
    meta["note"] = note
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))


# ── Progression ───────────────────────────────────────────────────────────────
kept = {c: sum(1 for name, t in meta["takes"].items()
               if t.get("condition") == c and t.get("keep")) for c, _ in CAMPAGNE}
total_kept = sum(kept.values())
st.progress(min(1.0, total_kept / (len(CAMPAGNE) * takes_per_cond)),
            text=f"{total_kept}/{len(CAMPAGNE) * takes_per_cond} prises validées")

cols = st.columns(len(CAMPAGNE))
for col, (c, _) in zip(cols, CAMPAGNE):
    col.metric(c, f"{kept[c]}/{takes_per_cond}")

# Condition courante = première non complète
current = next(((c, desc) for c, desc in CAMPAGNE if kept[c] < takes_per_cond), None)

if current is None:
    st.success("🎉 Campagne complète ! La session est prête pour une recette "
               "(source `studio`).")
    st.stop()

cond, consigne = current
st.subheader(f"Condition : {cond} — prise {kept[cond] + 1}/{takes_per_cond}")
st.markdown(f"**Consigne : {consigne}.** Prononce « **éloquence** » une fois, "
            "naturellement, puis re-clique pour arrêter.")

# Le widget micro de Streamlit est discret : on le grossit en un vrai bouton
# d'enregistrement, impossible à rater.
st.markdown("""
<style>
div[data-testid="stAudioInput"] {
    border: 3px solid #C0392B; border-radius: 14px;
    padding: 18px; background: rgba(192, 57, 43, 0.06);
}
div[data-testid="stAudioInput"] button {
    transform: scale(1.7); margin: 10px 18px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("### 🔴 Enregistrer — clique le micro ci-dessous (start), re-clique (stop)")
audio = st.audio_input(f"Prise « {cond} »", key=f"rec_{cond}_{kept[cond]}")

if audio is not None:
    # Conversion navigateur (44.1/48 kHz, format variable) → 16 kHz mono PCM16.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio.getvalue())
        raw_path = Path(tmp.name)
    take_name = f"{cond}_{kept[cond] + 1:02d}.wav"
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path = session_dir / take_name
    conv = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
         "-ar", str(word.sample_rate), "-ac", "1", "-sample_fmt", "s16", str(out_path)],
        capture_output=True, text=True)
    raw_path.unlink(missing_ok=True)
    if conv.returncode != 0:
        st.error(f"Conversion échouée : {conv.stderr[-300:]}")
        st.stop()

    m = measure_clip(out_path, word.sample_rate)
    verdict, raisons = judge_clip(m, GATE, word.sample_rate, pool="studio")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Durée", f"{m.get('duration_s', 0):.2f} s")
    c2.metric("RMS", f"{m.get('rms', 0):.3f}")
    c3.metric("Pic", f"{m.get('peak', 0):.2f}")
    c4.metric("Saturation", f"{m.get('saturation_ratio', 0):.1%}")
    c5.metric("SNR estimé", f"{m.get('snr_db', 0):.0f} dB")
    st.audio(str(out_path))

    if verdict == "rejete":
        st.error("Prise rejetée par le contrôle qualité : " + " · ".join(raisons)
                 + " — réenregistre.")
        out_path.unlink(missing_ok=True)
    else:
        if verdict == "douteux":
            st.warning("Qualité limite : " + " · ".join(raisons)
                       + " — à toi de juger à la réécoute.")
        col_ok, col_ko = st.columns(2)
        if col_ok.button("✅ Garder cette prise", type="primary", key=f"k_{take_name}"):
            meta["takes"][take_name] = {
                "condition": cond, "keep": True, "verdict_auto": verdict,
                "raisons": raisons, "mesures": m,
                "saved_at": datetime.datetime.now().isoformat(timespec="seconds")}
            save_meta()
            st.rerun()
        if col_ko.button("🔁 Refaire", key=f"r_{take_name}"):
            out_path.unlink(missing_ok=True)
            st.rerun()

# ── Prises de la session ──────────────────────────────────────────────────────
if meta["takes"]:
    with st.expander(f"Prises validées ({total_kept})"):
        for name, t in sorted(meta["takes"].items()):
            if not t.get("keep"):
                continue
            left, right = st.columns([2, 4])
            left.markdown(f"`{name}` — {t['condition']}"
                          + (" ⚠️" if t.get("verdict_auto") == "douteux" else ""))
            wav = session_dir / name
            if wav.exists():
                right.audio(str(wav))
