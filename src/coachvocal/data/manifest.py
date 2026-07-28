"""Manifest = la liste EXACTE des fichiers d'un run, avec leur pondération.

C'est la pièce de traçabilité centrale : on ne verse jamais l'audio dans git,
mais on versionne le manifest. Deux runs comparables doivent avoir des manifests
comparables, et un écart inexpliqué de métriques commence toujours par un `diff`
de manifests.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

FIELDS = ("file", "label", "pool", "split", "copies")


@dataclass
class Manifest:
    rows: list[dict] = field(default_factory=list)

    def add(self, pool: str, files: list[Path], label: int, split: str, copies: int = 1) -> None:
        for f in files:
            self.rows.append({"file": str(f), "label": label, "pool": pool,
                              "split": split, "copies": copies})

    # ── Vues ──────────────────────────────────────────────────────────────────
    def paths_labels(self, split: str) -> tuple[list[str], list[int]]:
        """Chemins dupliqués selon `copies` (le « boost ») + labels alignés."""
        paths, labels = [], []
        for r in self.rows:
            if r["split"] == split:
                paths.extend([r["file"]] * r["copies"])
                labels.extend([r["label"]] * r["copies"])
        return paths, labels

    def files(self, split: str | None = None, label: int | None = None) -> list[Path]:
        return [Path(r["file"]) for r in self.rows
                if (split is None or r["split"] == split) and (label is None or r["label"] == label)]

    def composition(self) -> dict[str, dict[str, dict[str, int]]]:
        """{split: {pool: {clips, effectifs}}} — effectifs = après boost."""
        out: dict = defaultdict(lambda: defaultdict(lambda: {"clips": 0, "effective": 0}))
        for r in self.rows:
            cell = out[r["split"]][r["pool"]]
            cell["clips"] += 1
            cell["effective"] += r["copies"]
        return {s: dict(p) for s, p in out.items()}

    def balance(self) -> dict[str, dict[str, int]]:
        out: dict = defaultdict(Counter)
        for r in self.rows:
            out[r["split"]]["pos" if r["label"] == 1 else "neg"] += r["copies"]
        return {s: dict(c) for s, c in out.items()}

    def summary_lines(self) -> list[str]:
        lines = []
        for split, c in self.balance().items():
            pos, neg = c.get("pos", 0), c.get("neg", 0)
            lines.append(f"  {split:5} : {pos:6} positifs | {neg:6} négatifs "
                         f"(ratio 1:{neg / max(pos, 1):.1f})")
        return lines

    # ── Entrées / sorties ─────────────────────────────────────────────────────
    def to_csv(self, path: Path, relative_to: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in self.rows:
                row = dict(r)
                if relative_to:
                    try:
                        row["file"] = str(Path(r["file"]).relative_to(relative_to))
                    except ValueError:
                        pass
                w.writerow(row)

    @classmethod
    def from_csv(cls, path: Path) -> Manifest:
        with open(path, newline="") as f:
            rows = [{**r, "label": int(r["label"]), "copies": int(r["copies"])}
                    for r in csv.DictReader(f)]
        return cls(rows=rows)

    def fingerprint(self) -> str:
        """Empreinte du jeu de données (indépendante de l'ordre).

        Deux runs de même empreinte ont vu exactement les mêmes fichiers avec les
        mêmes poids — c'est ce qui permet d'affirmer « seule la seed a changé »."""
        key = sorted(f"{r['split']}|{r['pool']}|{r['label']}|{r['copies']}|{r['file']}"
                     for r in self.rows)
        return hashlib.sha256("\n".join(key).encode()).hexdigest()[:16]
