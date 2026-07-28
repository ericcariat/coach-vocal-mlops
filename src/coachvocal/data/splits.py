"""Split train/val/test — **figé une fois, jamais re-tiré**.

Deux règles qui décident de la crédibilité de toutes les métriques du projet :

1. **Split par GROUPE, pas par clip.** Un groupe = une vidéo YouTube source ou
   une session d'enregistrement. Deux extraits de la même vidéo partagent le
   locuteur, le micro et le bruit de fond : séparés au hasard, ils font monter
   le score sans que le modèle ait rien appris de généralisable.
2. **Écrit une seule fois.** Re-tirer le split à chaque run, c'est finir par
   choisir (sans le vouloir) le tirage qui donne le plus beau chiffre. La
   commande refuse d'écraser un `splits.csv` existant sans `--force`.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

RATIO = (0.8, 0.1, 0.1)
FIELDS = ("file", "label", "source", "group", "split")


def group_of(filename: str) -> tuple[str, str]:
    """(source, groupe) déduits du nom de fichier.

    `yt_<video_id>_...` → YouTube, groupe = la vidéo.
    `moi_...`           → mes enregistrements, groupe = la session (préfixe date/heure).
    """
    if filename.startswith("yt_"):
        m = re.match(r"(yt_[A-Za-z0-9_-]{11})", filename)
        return "youtube", (m.group(1) if m else filename)
    if filename.startswith("moi_"):
        m = re.match(r"(moi_[0-9]{4,8})", filename)
        return "moi", (m.group(1) if m else "moi")
    return "autre", filename.split("_")[0]


def assign(group: str, ratio: tuple[float, float, float] = RATIO, seed: int = 42) -> str:
    """Affectation déterministe par hash du groupe : reproductible partout, sans
    état, et stable si de nouveaux clips arrivent (un groupe existant ne change
    jamais de split)."""
    h = int(hashlib.md5(f"{seed}:{group}".encode()).hexdigest(), 16) % 1000 / 1000
    if h < ratio[0]:
        return "train"
    return "val" if h < ratio[0] + ratio[1] else "test"


def freeze(word_dir: Path, wakeword: str, seed: int = 42, force: bool = False,
           ratio: tuple[float, float, float] = RATIO) -> Path:
    """Construit `splits.csv` depuis `clean/positives` et `clean/negatives_proches`."""
    out = word_dir / "splits.csv"
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} existe déjà — le split est FIGÉ par principe. "
            "Le refaire invaliderait la comparabilité de tous les runs précédents "
            "(--force si tu sais ce que tu fais).")

    rows = []
    for sub, label in (("positives", wakeword), ("negatives_proches", "proche")):
        for f in sorted((word_dir / "clean" / sub).glob("*.wav")):
            source, group = group_of(f.name)
            rows.append({"file": f.name, "label": label, "source": source,
                         "group": group, "split": assign(group, ratio, seed)})

    if not rows:
        raise RuntimeError(f"aucun clip trouvé sous {word_dir / 'clean'}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


def summarize(splits_csv: Path) -> dict:
    counts: dict = defaultdict(lambda: defaultdict(int))
    groups: dict = defaultdict(set)
    with open(splits_csv, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["split"]][row["label"]] += 1
            groups[row["split"]].add(row["group"])
    return {s: {"clips": dict(c), "groups": len(groups[s])} for s, c in counts.items()}
