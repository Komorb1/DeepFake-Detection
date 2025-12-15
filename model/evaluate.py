import sys
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "processed"
DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "checkpoints" / "ffpp_best_20251214_000015_ft80_lr5e-06.keras"


def _get_frame_indices(T: int, avg_frames: int):
    if avg_frames == 1:
        return [T // 2]
    if avg_frames == 3:
        return [T // 4, T // 2, (3 * T) // 4]
    return [T // 6, T // 3, T // 2, (2 * T) // 3, (5 * T) // 6]


def _load_npz(processed_dir: Path, dataset: str):
    npz_path = processed_dir / f"{dataset}_rgb_dct_splits.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")
    return np.load(npz_path, allow_pickle=True)


def _load_dct_stats(processed_dir: Path, stats_dataset: str):
    stats = _load_npz(processed_dir, stats_dataset)
    if "dct_train_min" not in stats or "dct_train_max" not in stats:
        raise KeyError(
            f"{stats_dataset}_rgb_dct_splits.npz is missing dct_train_min/dct_train_max. "
            f"Re-run preprocessing that saves these keys."
        )
    return float(stats["dct_train_min"]), float(stats["dct_train_max"])


def _normalize_dct(dct_raw: np.ndarray, dmin: float, dmax: float, eps: float = 1e-8):
    return (dct_raw - dmin) / ((dmax - dmin) + eps)


def pack_split(data, split: str, dct_stats, clip_dct: bool):
    """
    split in {"val","test"} (train not needed for eval)
    Returns:
      X_rgb_seq: (N,T,H,W,3) float32 in [0,1]
      X_dct_vec: (N,4096) float32
      y:         (N,) int
    """
    X_rgb = data[f"X_{split}_rgb"].astype(np.float32)
    y = data[f"y_{split}"].astype(int)

    raw_key = f"X_{split}_dct_raw"
    norm_key = f"X_{split}_dct"

    dmin, dmax = dct_stats
    if raw_key in data:
        X_dct = _normalize_dct(data[raw_key].astype(np.float32), dmin, dmax)
    else:
        # fallback (older npz)
        print(f"[WARN] {raw_key} not found; using {norm_key} as-is (may be unfair cross-dataset).")
        X_dct = data[norm_key].astype(np.float32)

    if clip_dct:
        X_dct = np.clip(X_dct, 0.0, 1.0)

    X_dct_vec = X_dct.mean(axis=1).reshape(len(X_dct), -1).astype(np.float32)
    return X_rgb, X_dct_vec, y


def predict_probs_single_frame(
    model,
    X_img,
    X_dct_vec,
    img_size: int,
    batch_size: int,
    dct_mode: str,
):
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    ds = tf.data.Dataset.from_tensor_slices((X_img, X_dct_vec))

    def _map(rgb, dct):
        rgb = tf.cast(rgb, tf.float32)  # 0..1
        rgb = tf.image.resize(rgb, (img_size, img_size))
        dct = tf.cast(dct, tf.float32)

        if dct_mode == "rgb_only":
            dct = tf.zeros_like(dct)
        elif dct_mode == "dct_only":
            rgb = tf.zeros_like(rgb)

        rgb = preprocess_input(rgb * 255.0)
        return {"rgb_input": rgb, "dct_input": dct}

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    probs = model.predict(ds, verbose=0).squeeze()
    return np.array(probs)


def predict_probs_frame_averaging(
    model,
    X_rgb_seq,
    X_dct_vec,
    img_size: int,
    batch_size: int,
    frame_indices,
    dct_mode: str,
):
    probs_all = []
    for idx in frame_indices:
        probs_all.append(
            predict_probs_single_frame(
                model,
                X_rgb_seq[:, idx],
                X_dct_vec,
                img_size,
                batch_size,
                dct_mode=dct_mode,
            )
        )
    return np.stack(probs_all, axis=0).mean(axis=0)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "balanced_acc"):
    y_true = y_true.astype(int)
    n_real = int((y_true == 0).sum())
    n_fake = int((y_true == 1).sum())

    best_tau, best_score = 0.5, -1.0
    for tau in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= tau).astype(int)

        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        real_rec = tn / max(n_real, 1)     # TNR
        fake_rec = tp / max(n_fake, 1)     # TPR
        bal_acc = 0.5 * (real_rec + fake_rec)

        prec_fake = tp / max(tp + fp, 1)
        f1 = (2 * prec_fake * fake_rec) / max(prec_fake + fake_rec, 1e-12)

        score = bal_acc if objective == "balanced_acc" else f1
        if score > best_score:
            best_score = score
            best_tau = float(tau)

    return best_tau, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ffpp", choices=["ffpp", "celebdf"])
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processed_dir", type=str, default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--avg_frames", type=int, default=3, choices=[1, 3, 5])

    parser.add_argument(
        "--threshold",
        type=str,
        default="0.50",
        help="Float threshold like 0.55, or 'auto' to select on VAL of --threshold_from",
    )
    parser.add_argument("--objective", type=str, default="balanced_acc", choices=["balanced_acc", "f1"])

    parser.add_argument(
        "--dct_stats_from",
        type=str,
        default="self",
        choices=["self", "ffpp", "celebdf"],
        help="Which dataset's TRAIN DCT min/max to use. For FF++->CelebDF, use ffpp.",
    )
    parser.add_argument(
        "--threshold_from",
        type=str,
        default="self",
        choices=["self", "ffpp", "celebdf"],
        help="If --threshold auto, pick tau using VAL from this dataset (use ffpp to avoid peeking).",
    )

    parser.add_argument(
        "--dct_mode",
        type=str,
        default="both",
        choices=["both", "rgb_only", "dct_only"],
        help="Use both inputs, or ablate one branch by zeroing it.",
    )
    parser.add_argument(
        "--clip_dct",
        action="store_true",
        help="Clip normalized DCT to [0,1] (recommended cross-dataset).",
    )

    args = parser.parse_args()
    processed_dir = Path(args.processed_dir)

    model_path = Path(args.model_path.strip())
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path))

    dct_stats_dataset = args.dataset if args.dct_stats_from == "self" else args.dct_stats_from
    dmin, dmax = _load_dct_stats(processed_dir, dct_stats_dataset)
    dct_stats = (dmin, dmax)
    print(f"DCT stats source: {dct_stats_dataset} (train_min={dmin:.6f}, train_max={dmax:.6f})")
    print(f"DCT mode: {args.dct_mode} | clip_dct={args.clip_dct}")

    eval_data = _load_npz(processed_dir, args.dataset)
    X_val_rgb_seq, X_val_dct_vec, y_val = pack_split(eval_data, "val", dct_stats=dct_stats, clip_dct=args.clip_dct)
    X_test_rgb_seq, X_test_dct_vec, y_test = pack_split(eval_data, "test", dct_stats=dct_stats, clip_dct=args.clip_dct)

    T = X_test_rgb_seq.shape[1]
    frame_indices = _get_frame_indices(T, args.avg_frames)

    if args.threshold.strip().lower() == "auto":
        thr_dataset = args.dataset if args.threshold_from == "self" else args.threshold_from
        thr_data = _load_npz(processed_dir, thr_dataset)
        X_thr_val_rgb_seq, X_thr_val_dct_vec, y_thr_val = pack_split(
            thr_data, "val", dct_stats=dct_stats, clip_dct=args.clip_dct
        )

        val_probs = predict_probs_frame_averaging(
            model,
            X_thr_val_rgb_seq,
            X_thr_val_dct_vec,
            img_size=args.img_size,
            batch_size=args.batch_size,
            frame_indices=frame_indices,
            dct_mode=args.dct_mode,
        )

        tau, best_score = find_best_threshold(y_thr_val, val_probs, objective=args.objective)
        print(f"Auto threshold selected on VAL of {thr_dataset}: τ={tau:.2f} (best {args.objective}={best_score:.3f})")
    else:
        tau = float(args.threshold)

    test_probs = predict_probs_frame_averaging(
        model,
        X_test_rgb_seq,
        X_test_dct_vec,
        img_size=args.img_size,
        batch_size=args.batch_size,
        frame_indices=frame_indices,
        dct_mode=args.dct_mode,
    )

    y_true = y_test.astype(int)
    y_pred = (test_probs >= tau).astype(int)

    print(f"\nModel: {model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Frames used: {frame_indices} (avg_frames={args.avg_frames})")
    print(f"Threshold τ = {tau:.2f}")

    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0
    ))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    if __package__ is None:
        sys.path.append(str(REPO_ROOT))
    main()
