"""Garde-fou anti-fuite SUMM-RE : banc, val_ambient et train DISJOINTS.

Les trois usages du corpus de réunions doivent rester étanches, par RÉUNION et
par LOCUTEUR (un même locuteur en train et au banc mesurerait de la
mémorisation de voix). La convention de nommage `summre_<réunion>_<xxxx>_<locuteur>.wav`
porte l'information — ce test la vérifie sur les fichiers réellement présents.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRS = {
    "banc": ROOT / "data/external/bench_extra",
    "val_ambient": ROOT / "data/wakewords/eloquence/val_ambient",
    "train": ROOT / "data/external/summre_train",
}


def _ids(d: Path) -> tuple[set, set]:
    meetings, speakers = set(), set()
    for f in d.glob("summre_*.wav"):
        m = re.match(r"summre_(\w{4})_\w{4}_(\d+)", f.stem)
        assert m, f"nom hors convention : {f.name}"
        meetings.add(m.group(1))
        speakers.add(m.group(2))
    return meetings, speakers


@pytest.mark.skipif(not all(d.exists() for d in DIRS.values()),
                    reason="corpus SUMM-RE absent (dépôt allégé)")
def test_reunions_et_locuteurs_disjoints():
    ids = {name: _ids(d) for name, d in DIRS.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (ids[a][0] & ids[b][0]), \
                f"réunion(s) partagée(s) entre {a} et {b} : {ids[a][0] & ids[b][0]}"
            assert not (ids[a][1] & ids[b][1]), \
                f"locuteur(s) partagé(s) entre {a} et {b} : {ids[a][1] & ids[b][1]}"


@pytest.mark.skipif(not DIRS["train"].exists(), reason="summre_train absent")
def test_train_non_vide_si_recette_v15():
    assert list(DIRS["train"].glob("summre_*.wav")), \
        "speech_neg_300_summre référencé mais summre_train/ vide"
