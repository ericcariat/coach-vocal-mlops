"""Rappel par forme au banc : regroupement des surfaces et appariement."""

from coachvocal.data.corpus import surface_form
from coachvocal.evaluation.stream_bench import score


def test_surface_form_regroupe():
    w = "eloquence"
    assert surface_form("éloquence", w) == "nu"
    assert surface_form("Éloquence", w) == "nu"
    assert surface_form("l'éloquence", w) == "l'"
    assert surface_form("L’éloquence", w) == "l'"     # apostrophe typographique
    assert surface_form("d'éloquence", w) == "d'"
    assert surface_form("Dauphine-Éloquence", w) == "autre"


def test_score_renvoie_les_hits_par_occurrence():
    # 2 occurrences, 1 seul trigger apparié à la première
    sc = score(triggers=[10.2], occurrences=[10.0, 30.0], uncertain=[])
    assert sc["hits"] == [True, False]
    assert sc["detected"] == 1 and sc["n_occ"] == 2
    kinds = [e["kind"] for e in sc["events"]]
    assert kinds.count("FN") == 1 and kinds.count("TP") == 1
