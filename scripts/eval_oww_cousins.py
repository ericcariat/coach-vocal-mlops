"""Banc des cousins — le point faible mesurable des têtes openWakeWord.

Le banc streaming n'a presque pas de cousins phonétiques (« éloquente »,
« élégance »…) : la faiblesse du round 5 ne s'y voyait pas, elle se voyait au
micro. Ici on mesure la SÉPARATION de la tête sur des clips ciblés :

  - positifs moi_    : doivent déclencher (haut)         [vus à l'entraînement]
  - cousins moi_     : ne doivent PAS déclencher (bas)   [vus si --adv-weight]
  - cousins TTS      : ne doivent pas déclencher          [jamais vus]
  - hard negatives   : ne doivent pas déclencher          [vus]

Les groupes « vus à l'entraînement » mesurent la mémorisation utile, pas la
généralisation — c'est écrit dans la sortie. Le juge final reste le test guidé
au micro de l'auteur. Sortie : tableau par seuil + PNG des distributions.

    uv run python scripts/eval_oww_cousins.py open_wake_word_compare/<tête>.onnx …
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as _np  # noqa: E402
import soundfile as _sf  # noqa: E402
from train_oww_head import PAD_S, SR, embed_arrays, embed_files  # noqa: E402

from coachvocal.evaluation.oww_adapter import OWW_DIR  # noqa: E402

THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.99]

GROUPES = [
    ("positifs moi_ (attendu : HAUT, vus)",
     sorted((ROOT / "data/wakewords/eloquence/clean/positives").glob("moi_*.wav")), True),
    ("cousins moi_ (attendu : BAS, vus si adv-weight)",
     sorted((ROOT / "data/wakewords/eloquence/clean/negatives_proches").glob("moi_*.wav")), False),
    ("cousins TTS (attendu : BAS, jamais vus)",
     sorted((ROOT / "data/wakewords/eloquence/generated/tts_neg_proches").glob("*.wav"))[:150], False),
    ("hard negatives banc (attendu : BAS, vus)",
     sorted((ROOT / "exports/oww_training_b/negatifs_adversariaux").glob("hn_*.wav")), False),
]


def main():
    heads = [Path(a) for a in sys.argv[1:]]
    if not heads:
        sys.exit("usage : eval_oww_cousins.py <tête.onnx> [...]")

    import onnxruntime as ort
    mel = ort.InferenceSession(str(OWW_DIR / "melspectrogram.onnx"),
                               providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(str(OWW_DIR / "embedding_model.onnx"),
                               providers=["CPUExecutionProvider"])

    feats = [(nom, embed_files(files, mel, emb, f"cousins_{i}"), attendu_haut)
             for i, (nom, files, attendu_haut) in enumerate(GROUPES) if files]

    # Préfixes tronqués du mot (fuite mesurée le 2026-08-14 : la tête tirait
    # dès 80-90 % du mot — « éloquen » déclenchait, « éloquente » passait).
    from coachvocal.data.sources.fragments import word_span
    pad_n = int(PAD_S * SR)
    for frac in (0.8, 0.9):
        arrs = []
        for f in sorted((ROOT / "data/wakewords/eloquence/clean/positives").glob("moi_*.wav")):
            a, sr = _sf.read(f, dtype="float32")
            if sr != SR:
                continue
            w0, w1 = word_span(a, SR)
            k = int(frac * (w1 - w0))
            arrs.append(_np.pad(a[w0:w0 + k], (pad_n - k, 0)).astype(_np.float32))
        feats.append((f"préfixes {frac:.0%} du mot (attendu : BAS)",
                      embed_arrays(arrs, mel, emb, f"prefix{int(frac * 100)}"), False))

    import matplotlib
    matplotlib.use("Agg")
    # Les taux sont PERSISTÉS (JSON, fusion par tête) : la page Évaluation
    # affiche la section « Ma voix » depuis ce fichier.
    import datetime
    import json

    import matplotlib.pyplot as plt
    json_out = ROOT / "artifacts/reports/oww_cousins.json"
    store = json.loads(json_out.read_text()) if json_out.exists() else {}

    fig, axes = plt.subplots(len(heads), 1, figsize=(9, 3.2 * len(heads)),
                             squeeze=False)
    for hi, head_path in enumerate(heads):
        # Une tête du registre s'appelle model.onnx : on prend le nom du run.
        label = head_path.stem if head_path.stem != "model" else head_path.parent.name
        sess = ort.InferenceSession(str(head_path),
                                    providers=["CPUExecutionProvider"])
        print(f"\n=== {label} ===")
        ax = axes[hi][0]
        entry = {"date": datetime.date.today().isoformat(), "groupes": {}}
        for gi, (nom, X, attendu_haut) in enumerate(feats):
            p = np.concatenate([sess.run(None, {"input": X[k:k + 1].astype(np.float32)})[0].ravel()
                                for k in range(len(X))])
            taux = " · ".join(f"@{t}: {(p > t).mean():.0%}" for t in THRESHOLDS)
            print(f"  {nom} ({len(p)}) — {taux}")
            entry["groupes"][nom] = {
                "n": int(len(p)), "attendu": "haut" if attendu_haut else "bas",
                "taux": {str(t): round(float((p > t).mean()), 4) for t in THRESHOLDS}}
            ax.hist(p, bins=40, range=(0, 1), alpha=0.55, label=f"{nom} (n={len(p)})")
            _ = gi
        store[label] = entry
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("probabilité de la tête")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = ROOT / "artifacts/reports/oww_cousins.png"
    fig.savefig(out, dpi=120)
    json_out.write_text(json.dumps(store, indent=1, ensure_ascii=False))
    print(f"\n💾  {out}\n💾  {json_out}")


if __name__ == "__main__":
    main()
