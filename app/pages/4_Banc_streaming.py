"""Page Banc streaming — la mesure qui décide, et l'audit des erreurs à l'oreille."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coachvocal import paths, registry  # noqa: E402

st.set_page_config(page_title="Banc streaming", page_icon="🎬", layout="wide")
st.title("🎬 Banc streaming")

WAKEWORD = "eloquence"
bench = registry.bench_results(WAKEWORD)

st.markdown("""
On rejoue la **logique live exacte** (fenêtre d'1 s, décision toutes les 125 ms,
3 fenêtres consécutives, cooldown) sur de l'audio YouTube **continu jamais vu à
l'entraînement**, avec pour vérité terrain les alignements WhisperX.

Deux chiffres décrivent l'expérience réelle : le **rappel streaming** (occurrences
réellement attrapées) et les **fausses alarmes par heure**.
""")

if not bench.get("results"):
    st.info("Aucun banc enregistré.")
    st.code("uv run coachvocal bench --run v03_replica --minutes 16", language="bash")
    st.stop()

c = st.columns(3)
c[0].metric("Audio analysé", f"{bench['total_seconds'] / 60:.1f} min")
c[1].metric("Occurrences", bench.get("n_occurrences", "—"))
c[2].metric("Vidéos exclues (vues à l'entraînement)", len(bench.get("forbidden_videos", [])))

rows = []
for model, per_th in bench["results"].items():
    for th, r in per_th.items():
        rows.append({"Modèle": model, "Seuil": th.replace("th", ""),
                     "Rappel": r["recall_stream"], "Détectées": f"{r['detected']}/{r['n_occ']}",
                     "Fausses alarmes": r["false_alarms"], "FA / heure": r["fa_per_hour"],
                     "Incertains": r["uncertain"]})
st.dataframe(pd.DataFrame(rows).style.format({"Rappel": "{:.1%}", "FA / heure": "{:.1f}"}),
             width="stretch", hide_index=True)

st.info("""**Lire les FA/heure honnêtement** : le corpus est thématique (des vidéos
où l'on *parle* d'éloquence), c'est un pire cas volontaire et non une moyenne de
la vie courante. Les événements « incertains » — vérité terrain douteuse — sont
exclus du décompte plutôt que comptés à charge.""")

# ── Audit des erreurs ─────────────────────────────────────────────────────────
st.subheader("Erreurs à écouter")
model = st.selectbox("Modèle", list(bench["results"]))
th = st.selectbox("Seuil", list(bench["results"][model]))
events = bench["results"][model][th].get("events", [])
if not events:
    st.caption("Aucun événement enregistré (banc lancé sans collecte).")
else:
    kinds = st.multiselect("Type", ["FA", "FN", "INCERTAIN"], default=["FA", "FN"])
    filtered = [e for e in events if e["kind"] in kinds]

    # ── Verdicts persistés (jugement à l'oreille) ────────────────────────────
    # Un fichier JSON à côté des résultats du banc : une entrée par événement,
    # avec le verdict et sa date. C'est la trace qui alimente les hard negatives.
    verdicts_file = paths.report_dir("stream_bench") / f"{WAKEWORD}_verdicts.json"
    verdicts: dict = json.loads(verdicts_file.read_text()) if verdicts_file.exists() else {}

    A_JUGER = "— à juger —"
    CONFIRME = "✅ Erreur confirmée"
    PAS_ERREUR = "❌ Pas une erreur (vérité terrain à revoir)"
    # Inexploitable = audio inutilisable à l'oreille (mot coupé, hurlé, bouillie) :
    # l'événement est à exclure du banc — ni compté à charge, ni hard negative.
    INEXPLOITABLE = "🗑️ Inexploitable (à exclure du banc)"
    OPTIONS = [A_JUGER, CONFIRME, PAS_ERREUR, INEXPLOITABLE]

    def event_key(e: dict) -> str:
        return f"{model}|{th}|{e['segment']}|{e['t']}|{e['kind']}"

    # Sauvegarde AVANT l'affichage du tableau : les widgets radio (plus bas)
    # gardent leur état dans st.session_state entre deux exécutions du script.
    changed = False
    for e in filtered:
        key = event_key(e)
        choice = st.session_state.get(f"verdict_{key}")
        if choice is None:
            continue
        saved = verdicts.get(key, {}).get("verdict", A_JUGER)
        if choice != saved:
            if choice == A_JUGER:
                verdicts.pop(key, None)
            else:
                verdicts[key] = {"verdict": choice,
                                 "saved_at": datetime.date.today().isoformat()}
            changed = True
    if changed:
        verdicts_file.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))

    # ── Tableau récapitulatif, verdict enregistré inclus ─────────────────────
    table = []
    for e in filtered:
        v = verdicts.get(event_key(e), {})
        table.append({"Type": e["kind"], "Instant (s)": e["t"],
                      "Segment": e["segment"], "Vidéo": e["video"],
                      "Verdict enregistré": v.get("verdict", A_JUGER),
                      "Enregistré le": v.get("saved_at", "")})
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    st.markdown("""
**Définitions**
- **FN — faux négatif (raté)** : le mot « éloquence » a réellement été prononcé
  (vérité WhisperX) mais le détecteur ne s'est pas déclenché.
- **FA — fausse alarme** : le détecteur s'est déclenché alors que le mot n'a pas
  été prononcé.
- **INCERTAIN** : vérité terrain douteuse (alignement WhisperX peu fiable à cet
  instant) — exclu du décompte des métriques plutôt que compté à charge.
""")

    st.caption(f"{len(filtered)} événement(s). Une erreur de détection se juge à "
               "l'oreille, pas sur un score : les fausses alarmes confirmées "
               "deviennent les *hard negatives* du prochain run.")

    # ── Écouter et juger ─────────────────────────────────────────────────────
    st.subheader("Écouter et juger")
    st.caption("Extrait de 2 s avant à 3 s après l'instant de l'événement. Le "
               "verdict est sauvegardé immédiatement ; pour l'annuler, revenir "
               "à « à juger ».")

    segment_paths = {Path(s["wav"]).name: s["wav"] for s in bench.get("segments", [])}

    @st.cache_data
    def audio_slice(wav_path: str, t: float):
        """Tranche [t−2 s, t+3 s] du segment, prête pour st.audio."""
        info = sf.info(wav_path)
        start = max(0, int((t - 2.0) * info.samplerate))
        stop = min(info.frames, int((t + 3.0) * info.samplerate))
        data, sr = sf.read(wav_path, start=start, stop=stop, dtype="float32")
        return data, sr

    # Les événements sont regroupés par type pour juger en série, toujours avec
    # la même question — le verdict stocké reste le même code (CONFIRME /
    # PAS_ERREUR), seul l'affichage change selon le type.
    GROUPES = {
        "FN": ("🔇 Faux négatifs — le détecteur n'a rien vu",
               "Vérifie que le mot **« éloquence » a bien été prononcé** dans "
               "chaque extrait. S'il l'est, le raté est confirmé.",
               {CONFIRME: "✅ Prononcé → raté confirmé",
                PAS_ERREUR: "❌ Pas prononcé → vérité terrain à corriger"}),
        "FA": ("🚨 Fausses alarmes — le détecteur s'est déclenché",
               "Vérifie que le mot **« éloquence » n'a PAS été prononcé** dans "
               "chaque extrait. S'il est absent, la fausse alarme est confirmée "
               "(→ futur *hard negative*).",
               {CONFIRME: "✅ Pas prononcé → FA confirmée",
                PAS_ERREUR: "❌ Prononcé → vérité terrain à corriger"}),
        "INCERTAIN": ("❓ Incertains — vérité terrain douteuse",
                      "Tranche : le mot **« éloquence » est-il clairement "
                      "prononcé** dans l'extrait ?",
                      {CONFIRME: "✅ Prononcé",
                       PAS_ERREUR: "❌ Pas prononcé"}),
    }

    for kind, (titre, consigne, labels) in GROUPES.items():
        groupe = [e for e in filtered if e["kind"] == kind]
        if not groupe:
            continue
        st.markdown(f"### {titre}")
        st.markdown(f"{consigne} *({len(groupe)} extrait(s))*")
        for e in groupe:
            key = event_key(e)
            saved = verdicts.get(key, {})
            left, mid, right = st.columns([2, 3, 3], vertical_alignment="center")
            with left:
                st.markdown(f"**{e['kind']}** à `{e['t']:.2f} s`")
                st.caption(f"{e['segment']}")
            with mid:
                wav = segment_paths.get(e["segment"])
                if wav and Path(wav).exists():
                    data, sr = audio_slice(wav, e["t"])
                    st.audio(data, sample_rate=sr)
                else:
                    st.warning("Audio source introuvable.")
            with right:
                st.radio("Verdict", OPTIONS,
                         index=OPTIONS.index(saved.get("verdict", A_JUGER)),
                         format_func=lambda o, _l=labels: _l.get(o, o),
                         key=f"verdict_{key}", horizontal=True,
                         label_visibility="collapsed")
                if saved:
                    st.caption(f"Enregistré le {saved['saved_at']}")
            st.divider()
