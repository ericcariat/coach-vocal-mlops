"""Corpus YouTube continu : segments audio + vérité terrain.

Ce corpus (produit par le scraper : `<video_id>_<debut>-<fin>.wav` + sous-titres
VTT + `discovery.db` avec les alignements WhisperX au mot) sert à deux choses :
- **le banc streaming**, seul juge crédible du détecteur (le test par clips
  classe mal les modèles — cf. docs/JOURNAL.md) ;
- **les négatifs de parole continue**, déficit n°1 identifié par ce même banc.

Deux référentiels de temps cohabitent, et les confondre a produit un premier
banc entièrement faux (rappel ~0 %) :
- **discovery.db** : temps WhisperX, alignés sur le CONTENU réel du fichier.
- **VTT** : sous-titres automatiques YouTube, qui dérivent de ±1 s ET ignorent
  le padding de tête de `yt-dlp --download-sections` (découpe aux keyframes :
  un segment nominal de 74 s peut faire 84 s de fichier).

→ La DB fait foi. Le VTT ne sert qu'à définir des **zones incertaines** : un
déclenchement non apparié qui tombe près d'un temps VTT n'est compté ni comme
détection ni comme fausse alarme (occurrence possiblement ratée par WhisperX).
"""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .. import paths

# Le corpus est volumineux et partagé avec le scraper : on le référence sans le
# copier. Surchargeable pour Docker/CI.
CORPUS = Path(os.environ.get("COACHVOCAL_YT_CORPUS", paths.EXTERNAL / "youtube_corpus"))

TS = r"(\d+):(\d+):(\d+\.\d+)"
VIDEO_ID = r"[A-Za-z0-9_-]{11}"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _to_s(h, m, s) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass
class Segment:
    wav: Path
    video_id: str
    start: int                      # début nominal dans la vidéo (nom de fichier)
    end: int
    vtt: Path | None = None
    occurrences: list[float] = None      # temps LOCAUX (s) des occurrences (DB)
    uncertain: list[float] = None        # temps LOCAUX (s) des zones VTT

    @property
    def duration(self) -> float:
        return float(self.end - self.start)


def db_occurrences(word: str, db: Path | None = None) -> dict[str, list[float]]:
    """video_id → temps absolus (référentiel du nom de fichier) du mot-clé."""
    db = db or CORPUS / "discovery.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = defaultdict(list)
    try:
        rows = con.execute(
            "SELECT video_id, t_start FROM clips WHERE word LIKE ?", (f"%{word}%",))
        for vid, t in rows:
            m = re.search(f"({VIDEO_ID})$", vid)     # certains video_id sont des URLs
            out[m.group(1) if m else vid].append(float(t))
    finally:
        con.close()
    return {k: sorted(v) for k, v in out.items()}


def vtt_occurrences(vtt: Path, stem: str) -> list[float]:
    """Instants des mots contenant `stem` dans un VTT YouTube mot-à-mot.

    Format : `<hh:mm:ss.mmm><c> mot</c>` ; le premier mot d'une cue ne porte pas
    de balise et hérite du début de cue. Dédupliqué à ±1 s (YouTube répète les
    lignes pour l'effet de défilement)."""
    times: list[float] = []
    cue_start = None
    for line in vtt.read_text(errors="ignore").splitlines():
        m = re.match(TS + r" --> ", line)
        if m:
            cue_start = _to_s(*m.groups())
            continue
        if cue_start is None or "<c>" not in line:
            continue
        t = cue_start
        for m2 in re.finditer(r"(?:<" + TS + r">)?<c>\s*([^<]+?)\s*</c>", line):
            if m2.group(1):
                t = _to_s(m2.group(1), m2.group(2), m2.group(3))
            if stem in strip_accents(m2.group(4)):
                times.append(t)
        head = line.split("<")[0]
        if stem in strip_accents(head):
            times.append(cue_start)
    times.sort()
    dedup: list[float] = []
    for t in times:
        if not dedup or t - dedup[-1] > 1.0:
            dedup.append(t)
    return dedup


def list_segments(word: str, corpus: Path | None = None,
                  uncertain_s: float = 5.0, require_vtt: bool = True) -> list[Segment]:
    """Tous les segments du corpus, annotés de leur vérité terrain locale."""
    corpus = corpus or CORPUS
    audio_dir, subs_dir = corpus / "audio", corpus / "subs"
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"corpus YouTube absent : {audio_dir}\n"
            "    → définir COACHVOCAL_YT_CORPUS, ou lier le dossier du scraper "
            "(cf. docs/DATA.md)")

    stem = strip_accents(word)[:7]
    db_occ = db_occurrences(word, corpus / "discovery.db")
    vtt_cache: dict[str, list[float]] = {}
    segments: list[Segment] = []

    for wav in sorted(audio_dir.glob("*.wav")):
        m = re.match(r"(.+)_(\d+)-(\d+)$", wav.stem)
        if not m:
            continue
        vid, s0, s1 = m.group(1), int(m.group(2)), int(m.group(3))
        vtt = next(iter(subs_dir.glob(f"{vid}.fr*.vtt")), None) if subs_dir.exists() else None
        if vtt is None and require_vtt:
            continue                      # pas de vérité terrain → inutilisable
        seg = Segment(wav=wav, video_id=vid, start=s0, end=s1, vtt=vtt)
        dur = seg.duration
        seg.occurrences = [t - s0 for t in db_occ.get(vid, []) if 0.5 <= t - s0 <= dur - 0.5]
        if vtt is not None:
            if vid not in vtt_cache:
                vtt_cache[vid] = vtt_occurrences(vtt, stem)
            seg.uncertain = [t - s0 for t in vtt_cache[vid]
                             if -uncertain_s <= t - s0 <= dur + uncertain_s]
        else:
            seg.uncertain = []
        segments.append(seg)
    return segments
