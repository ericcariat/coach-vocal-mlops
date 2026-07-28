"""Génération de positifs synthétiques avec Piper (TTS local).

Pourquoi : enregistrer 2000 « éloquence » à la main est impossible, et un
détecteur entraîné sur trois voix ne généralise pas. Piper tourne hors ligne, en
CPU, avec des voix françaises sous licence permissive, et son échantillonnage
stochastique fournit gratuitement de la diversité (deux appels identiques ne
donnent pas le même signal).

Trois leviers de diversité : la **voix** (modèle + locuteur), la **vitesse**
(`length_scale`) et la **variabilité** de synthèse (`noise_scale`). Le pool est
généré en couvrant toutes les combinaisons, ce qui permet ensuite un
échantillonnage stratifié à dose choisie.

Pièges rencontrés :
- Les voix entraînées sur des livres audio (`fr_FR-mls`) **babillent** sur un mot
  isolé : elles produisent 1,5 à 3,5 s de charabia au lieu du mot. Écartées.
- Ajouter un point final (« éloquence. ») stabilise nettement la prosodie.
- Un contrôle de durée après recadrage attrape les échecs restants : tout clip
  hors de [0,3 s ; 1,2 s] est régénéré, puis abandonné après N essais.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

QA_DUR_MIN_S = 0.30
QA_DUR_MAX_S = 1.20
QA_MAX_TRIES = 4


def piper_available() -> bool:
    return subprocess.run(["which", "uvx"], capture_output=True).returncode == 0


def _synthesize(text: str, model: Path, speaker: int, length_scale: float,
                noise_scale: float, out_dir: Path, n: int) -> list[Path]:
    """Appelle Piper en mode batch dans un environnement isolé (`uvx`), pour ne
    pas contaminer les dépendances figées du projet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join([text] * n))
        lines_file = f.name
    cmd = ["uvx", "--from", "piper-tts", "piper", "-m", str(model),
           "--speaker", str(speaker), "--length-scale", str(length_scale),
           "--noise-scale", str(noise_scale), "-i", lines_file,
           "-d", str(out_dir), "--output-dir-naming", "timestamp"]
    before = set(out_dir.glob("*.wav"))
    subprocess.run(cmd, check=True, capture_output=True)
    Path(lines_file).unlink(missing_ok=True)
    return sorted(set(out_dir.glob("*.wav")) - before)


def _postprocess(src: Path, dst: Path, sr: int, clip_samples: int) -> float | None:
    """22 050 Hz → 16 kHz, silences ôtés, recadrage centré 1 s, normalisation
    du pic. Renvoie la durée du mot après rognage (None si illisible)."""
    import librosa

    try:
        audio, _ = librosa.load(src, sr=sr, mono=True)
    except Exception:
        return None
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    duration = len(trimmed) / sr
    if len(trimmed) >= clip_samples:
        start = (len(trimmed) - clip_samples) // 2
        clip = trimmed[start:start + clip_samples]
    else:
        pad = clip_samples - len(trimmed)
        clip = np.pad(trimmed, (pad // 2, pad - pad // 2))
    peak = np.abs(clip).max()
    if peak > 0:
        clip = clip * (0.7 / peak)
    sf.write(dst, clip, sr, subtype="PCM_16")
    return duration


def generate_pool(word_dir: Path, text: str, voices: list[dict], length_scales: list[float],
                  noise_scales: list[float], per_combo: int, sr: int, clip_samples: int,
                  out_name: str = "tts_positives") -> dict:
    """Génère le pool complet + `manifest.csv` (une ligne par clip, avec sa
    combinaison — c'est ce qui rend l'échantillonnage stratifié possible)."""
    out_dir = word_dir / "generated" / out_name
    raw_dir = out_dir / "_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, rejected = [], 0
    for voice in voices:
        model = Path(voice["model"]).expanduser()
        if not model.exists():
            raise FileNotFoundError(f"voix Piper absente : {model} (cf. docs/DATA.md)")
        for ls in length_scales:
            for ns in noise_scales:
                tag = f"{voice['name']}_s{voice.get('speaker', 0)}_ls{ls}_ns{ns}"
                produced, tries = 0, 0
                while produced < per_combo and tries < QA_MAX_TRIES:
                    tries += 1
                    need = per_combo - produced
                    wavs = _synthesize(text, model, voice.get("speaker", 0), ls, ns,
                                       raw_dir, need)
                    for wav in wavs:
                        name = f"{tag}_{produced:04d}.wav"
                        duration = _postprocess(wav, out_dir / name, sr, clip_samples)
                        wav.unlink(missing_ok=True)
                        if duration is None or not (QA_DUR_MIN_S <= duration <= QA_DUR_MAX_S):
                            (out_dir / name).unlink(missing_ok=True)
                            rejected += 1
                            continue
                        rows.append({"file": name, "voice": voice["name"],
                                     "speaker": voice.get("speaker", 0),
                                     "length_scale": ls, "noise_scale": ns,
                                     "word_duration_s": round(duration, 3)})
                        produced += 1
                print(f"    {tag:<44} {produced}/{per_combo}")

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "voice", "speaker", "length_scale",
                                          "noise_scale", "word_duration_s"])
        w.writeheader()
        w.writerows(rows)

    meta = {"date": datetime.now().isoformat(timespec="seconds"), "text": text,
            "voices": voices, "length_scales": length_scales, "noise_scales": noise_scales,
            "per_combo": per_combo, "sample_rate": sr, "n_clips": len(rows),
            "n_rejected_qa": rejected,
            "qa": {"min_s": QA_DUR_MIN_S, "max_s": QA_DUR_MAX_S, "max_tries": QA_MAX_TRIES},
            "engine": "piper-tts via uvx"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if raw_dir.exists():
        for leftover in raw_dir.glob("*"):
            leftover.unlink()
        raw_dir.rmdir()
    print(f"\n✅  {len(rows)} clips TTS · {rejected} rejetés par le contrôle de durée")
    return meta
