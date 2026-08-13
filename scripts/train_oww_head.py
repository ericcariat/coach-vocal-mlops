"""Piste B, exécutée en local : une tête openWakeWord entraînée sur NOS données.

Chaîne : nos clips (1 s, 16 kHz) → padding de tête à ~2 s (mot à la FIN, la
convention oWW) → LEUR front-end gelé (mel 32 → embeddings Google, ONNX) →
features [16, 96] → tête 64x3 entraînée ici (Keras, CPU, ADR-002) → export
ONNX dans open_wake_word_compare/ → verdict au banc via l'adaptateur.

Différence assumée avec le pipeline officiel : nos négatifs (~2,6 h de clips
de recette) au lieu de leurs ~31 000 h ACAV — le silence légendaire ne
viendra peut-être pas ; ce qu'on teste, c'est « leur représentation + nos
voix réelles » contre leur tout-synthétique (16-24 % de rappel au banc).

Usage : uv run python scripts/train_oww_head.py [--seeds 42 43 44]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coachvocal import runtime  # noqa: E402
from coachvocal.evaluation.oww_adapter import (  # noqa: E402
    EMB_STEP_FRAMES,
    EMB_WIN_FRAMES,
    HEAD_EMBEDDINGS,
    OWW_DIR,
)

SR = 16000
PAD_S = 2.02                     # assez d'audio pour 196 trames mel pleines
EXPORT_DIR = ROOT / "open_wake_word_compare"
CACHE = ROOT / "artifacts" / "cache" / "eloquence" / "oww_features"


def collect_files() -> tuple[list[Path], list[Path]]:
    """Positifs réels + négatifs de nos recettes (train uniquement)."""
    import csv

    pos = sorted((ROOT / "exports/oww_training_b/positifs_reels").glob("*.wav"))
    neg = sorted((ROOT / "exports/oww_training_b/negatifs_adversariaux").glob("*.wav"))
    neg += sorted((ROOT / "exports/oww_training_b/negatifs_parole_continue_fr").glob("*.wav"))
    # + les négatifs génériques du TRAIN de la recette champion (bruit, GSC,
    # CV, silence, fragments) — notre océan à nous, même s'il est petit
    manifest = ROOT / "artifacts/runs/eloquence/v17_stack/manifest.csv"
    seen = {f.name for f in neg}
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            p = Path(row["file"])
            if (row["split"] == "train" and row["label"] == "0"
                    and p.name not in seen and p.exists()):
                neg.append(p)
                seen.add(p.name)
    return pos, neg


def embed_files(files: list[Path], mel_sess, emb_sess, tag: str) -> np.ndarray:
    """Chaque clip → features [16, 96] (mot padé en FIN de fenêtre ~2 s).
    Cache .npy : l'extraction ne se refait pas d'un run à l'autre."""
    import soundfile as sf

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{tag}_{len(files)}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    pad_n = int(PAD_S * SR)
    out = np.zeros((len(files), HEAD_EMBEDDINGS, 96), np.float32)
    batch_audio, batch_idx = [], []

    def flush():
        if not batch_audio:
            return
        for k, audio in enumerate(batch_audio):
            mel = mel_sess.run(None, {"input": audio[np.newaxis, :]})[0]
            frames = np.squeeze(mel) / 10.0 + 2.0
            wins = np.stack([frames[j:j + EMB_WIN_FRAMES]
                             for j in range(0, len(frames) - EMB_WIN_FRAMES + 1,
                                            EMB_STEP_FRAMES)])
            embs = emb_sess.run(None, {"input_1": wins[:, :, :, np.newaxis]})[0]
            out[batch_idx[k]] = embs.reshape(len(embs), 96)[-HEAD_EMBEDDINGS:]
        batch_audio.clear()
        batch_idx.clear()

    for i, f in enumerate(files):
        try:
            audio, sr = sf.read(f, dtype="float32")
        except Exception:
            continue
        if sr != SR:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[-pad_n:] if len(audio) >= pad_n else np.pad(
            audio, (pad_n - len(audio), 0))          # mot à la FIN
        batch_audio.append(audio.astype(np.float32))
        batch_idx.append(i)
        if len(batch_audio) >= 64:
            flush()
        if i % 1000 == 0:
            print(f"    {tag} : {i}/{len(files)}", flush=True)
    flush()
    np.save(cache_file, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    import onnxruntime as ort
    mel = ort.InferenceSession(str(OWW_DIR / "melspectrogram.onnx"),
                               providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(str(OWW_DIR / "embedding_model.onnx"),
                               providers=["CPUExecutionProvider"])

    pos_files, neg_files = collect_files()
    print(f"📦  {len(pos_files)} positifs réels · {len(neg_files)} négatifs (recette)")
    X_pos = embed_files(pos_files, mel, emb, "pos")
    X_neg = embed_files(neg_files, mel, emb, "neg")
    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(np.float32)
    print(f"    features : {X.shape}")

    runtime.configure(use_gpu=False)
    import tensorflow as tf
    from tensorflow import keras

    EXPORT_DIR.mkdir(exist_ok=True)
    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_val = int(0.1 * len(X))
        va, tr = idx[:n_val], idx[n_val:]

        head = keras.Sequential([
            keras.layers.Input(shape=(HEAD_EMBEDDINGS, 96)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        head.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss="binary_crossentropy",
                     metrics=[keras.metrics.AUC(name="auc")])
        n_pos, n_neg = y[tr].sum(), len(tr) - y[tr].sum()
        head.fit(X[tr], y[tr], validation_data=(X[va], y[va]),
                 epochs=args.epochs, batch_size=128, verbose=2,
                 class_weight={0: len(tr) / (2 * n_neg), 1: len(tr) / (2 * n_pos)},
                 callbacks=[keras.callbacks.EarlyStopping(
                     monitor="val_auc", mode="max", patience=6,
                     restore_best_weights=True)])

        out = EXPORT_DIR / f"eloquence_nosdonnees_64x3_seed{seed}.onnx"
        import tf2onnx
        spec = (tf.TensorSpec((1, HEAD_EMBEDDINGS, 96), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(head, input_signature=spec, opset=17,
                                   output_path=str(out))
        print(f"💾  {out.name}")

    print("\nBanc : uv run coachvocal bench --run open_wake_word_compare/"
          "eloquence_nosdonnees_64x3_seed42.onnx --minutes 60 --thresholds 0.05,0.2,0.5,0.8")


if __name__ == "__main__":
    main()
