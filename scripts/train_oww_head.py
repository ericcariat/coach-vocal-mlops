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


def acav_windows(path: str, stride: int = 8) -> np.ndarray:
    """Le fichier oWW est un FLUX d'embeddings (N, 96), un par 80 ms : on le
    découpe en fenêtres [16, 96] — les négatifs « océan » qui sculptent la
    frontière (l'ingrédient n°2 de leur silence)."""
    stream = np.load(path, mmap_mode="r")
    if stream.ndim == 3:                         # déjà fenêtré (N, 16, 96)
        return np.asarray(stream[::stride], dtype=np.float32)
    idx = np.arange(0, len(stream) - HEAD_EMBEDDINGS, stride)
    out = np.empty((len(idx), HEAD_EMBEDDINGS, 96), np.float32)
    for k, i in enumerate(idx):
        out[k] = stream[i:i + HEAD_EMBEDDINGS]
    return out



def collect_context_positives() -> list[np.ndarray]:
    """Positifs à CONTEXTE RÉEL (remède au round 2) : fenêtres de PAD_S s
    découpées dans les segments du corpus, fin du mot au bord droit, mot
    localisé par corrélation croisée avec le clip propre (la méthode
    auto-vérifiante de word_clips_recut). Les clips sans segment source
    (moi/guided) restent padés de silence — minoritaires."""
    import re

    import soundfile as sf

    from coachvocal.data import corpus as corpus_mod

    pad_n = int(PAD_S * SR)
    out: list[np.ndarray] = []
    seg_index = {}
    for wav in (corpus_mod.CORPUS / "audio").glob("*.wav"):
        m = re.match(r"(.+)_(\d+)-(\d+)$", wav.stem)
        if m:
            seg_index.setdefault(m.group(1), []).append(
                (int(m.group(2)), int(m.group(3)), wav))

    files = sorted((ROOT / "exports/oww_training_b/positifs_reels").glob("*.wav"))
    n_ctx = n_pad = 0
    for f in files:
        clip, csr = sf.read(f, dtype="float32")
        if csr != SR:
            continue
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        m = re.match(rf"yt_({corpus_mod.VIDEO_ID})_.*_(\d+(?:\.\d+)?)s", f.stem)
        placed = False
        if m:
            vid, t0 = m.group(1), float(m.group(2))
            for s0, s1, seg_wav in seg_index.get(vid, []):
                if not (s0 <= t0 <= s1):
                    continue
                audio, sr2 = sf.read(seg_wav, dtype="float32")
                if sr2 != SR:
                    break
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                expected = int((t0 - s0) * SR)
                lo = max(0, expected - 2 * SR)
                hi = min(len(audio), expected + 2 * SR + SR)
                template = clip[: SR // 2]
                zone = audio[lo:hi]
                if len(zone) < len(template) + 1:
                    break
                corr = np.correlate(zone, template, mode="valid")
                energy = np.convolve(zone ** 2, np.ones(len(template)), "valid")
                ncc = corr / (np.sqrt(energy * np.sum(template ** 2)) + 1e-9)
                if float(ncc.max()) < 0.6:
                    break
                w_start = lo + int(np.argmax(ncc))
                nz = (np.abs(clip) > 1e-5).nonzero()[0]
                w_len = int(nz[-1] + 1) if len(nz) else len(clip)
                end = min(len(audio), w_start + w_len + int(0.1 * SR))
                start = end - pad_n
                if start < 0:
                    break
                out.append(audio[start:end].astype(np.float32))
                n_ctx += 1
                placed = True
                break
        if not placed:                        # repli : silence en tête
            a = clip[-pad_n:] if len(clip) >= pad_n else np.pad(
                clip, (pad_n - len(clip), 0))
            out.append(a.astype(np.float32))
            n_pad += 1
    print(f"    positifs contexte réel : {n_ctx} · repli padding : {n_pad}")
    return out


def embed_arrays(arrays, mel_sess, emb_sess, tag: str) -> np.ndarray:
    """Comme embed_files mais depuis des tampons audio déjà chargés."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{tag}_{len(arrays)}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    out = np.zeros((len(arrays), HEAD_EMBEDDINGS, 96), np.float32)
    for i, audio in enumerate(arrays):
        mel = mel_sess.run(None, {"input": audio[np.newaxis, :]})[0]
        frames = np.squeeze(mel) / 10.0 + 2.0
        wins = np.stack([frames[j:j + EMB_WIN_FRAMES]
                         for j in range(0, len(frames) - EMB_WIN_FRAMES + 1,
                                        EMB_STEP_FRAMES)])
        embs = emb_sess.run(None, {"input_1": wins[:, :, :, np.newaxis]})[0]
        out[i] = embs.reshape(len(embs), 96)[-HEAD_EMBEDDINGS:]
        if i % 500 == 0:
            print(f"    {tag} : {i}/{len(arrays)}", flush=True)
    np.save(cache_file, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--acav", type=str, default=None,
                    help="features négatives oWW (.npy, flux N×96)")
    ap.add_argument("--acav-stride", type=int, default=8)
    ap.add_argument("--tag", type=str, default="nosdonnees")
    ap.add_argument("--context", action="store_true",
                    help="positifs à contexte réel (fenêtres 2 s du corpus)")
    args = ap.parse_args()

    import onnxruntime as ort
    mel = ort.InferenceSession(str(OWW_DIR / "melspectrogram.onnx"),
                               providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(str(OWW_DIR / "embedding_model.onnx"),
                               providers=["CPUExecutionProvider"])

    pos_files, neg_files = collect_files()
    print(f"📦  {len(pos_files)} positifs réels · {len(neg_files)} négatifs (recette)")
    if args.context:
        X_pos = embed_arrays(collect_context_positives(), mel, emb, "pos_ctx")
    else:
        X_pos = embed_files(pos_files, mel, emb, "pos")
    X_neg = embed_files(neg_files, mel, emb, "neg")
    parts_X, parts_y = [X_pos, X_neg], [np.ones(len(X_pos)), np.zeros(len(X_neg))]
    if args.acav:
        X_acav = acav_windows(args.acav, args.acav_stride)
        print(f"    + océan ACAV : {X_acav.shape} "
              f"({len(X_acav) * 0.08 * args.acav_stride / 3600:.1f} h de flux)")
        parts_X.append(X_acav)
        parts_y.append(np.zeros(len(X_acav)))
    X = np.concatenate(parts_X)
    y = np.concatenate(parts_y).astype(np.float32)
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

        out = EXPORT_DIR / f"eloquence_{args.tag}_64x3_seed{seed}.onnx"
        import tf2onnx

        # tf2onnx ne connaît pas les modèles Keras 3 : on exporte via une
        # tf.function qui enveloppe l'appel du modèle.
        spec = (tf.TensorSpec((1, HEAD_EMBEDDINGS, 96), tf.float32, name="input"),)

        @tf.function(input_signature=spec)
        def infer(x, _head=head):                 # lie la tête de CETTE itération
            return {"proba": _head(x, training=False)}

        tf2onnx.convert.from_function(infer, input_signature=spec, opset=17,
                                      output_path=str(out))
        # contrôle : l'ONNX doit rendre les mêmes probas que Keras
        import onnxruntime as _ort
        sess = _ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        probe = X[va[:8]].astype(np.float32)
        p_onnx = np.concatenate([sess.run(None, {"input": probe[k:k+1]})[0].ravel()
                                 for k in range(len(probe))])
        p_keras = head.predict(probe, verbose=0).ravel()
        assert np.allclose(p_onnx, p_keras, atol=1e-4), "export ONNX ≠ Keras !"
        print(f"💾  {out.name}  (export vérifié, écart max "
              f"{np.abs(p_onnx - p_keras).max():.2e})")

    print("\nBanc : uv run coachvocal bench --run open_wake_word_compare/"
          "eloquence_nosdonnees_64x3_seed42.onnx --minutes 60 --thresholds 0.05,0.2,0.5,0.8")


if __name__ == "__main__":
    main()
