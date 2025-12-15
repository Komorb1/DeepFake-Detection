import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from model.model import build_hybrid_model, set_finetune_layers

# (optional) reduce TensorFlow log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "processed"
CHECKPOINT_DIR = REPO_ROOT / "model" / "checkpoints"
SAVED_DIR = REPO_ROOT / "model" / "saved"


def load_npz(npz_path: Path):
    return np.load(npz_path, allow_pickle=True)


def make_inputs_from_npz(data):
    """
    Returns:
      X_*_rgb_seq: (N, T, H, W, 3) float32 in [0,1]
      X_*_img_center: (N, H, W, 3) center frame for training
      X_*_dct_vec: (N, 4096)
      y_*: (N,)
    """
    X_train_rgb = data["X_train_rgb"]
    X_val_rgb   = data["X_val_rgb"]
    X_test_rgb  = data["X_test_rgb"]

    X_train_dct = data["X_train_dct"]
    X_val_dct   = data["X_val_dct"]
    X_test_dct  = data["X_test_dct"]

    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    T = X_train_rgb.shape[1]
    center_idx = T // 2

    X_train_img_center = X_train_rgb[:, center_idx]
    X_val_img_center   = X_val_rgb[:, center_idx]
    X_test_img_center  = X_test_rgb[:, center_idx]

    X_train_dct_vec = X_train_dct.mean(axis=1).reshape(len(X_train_dct), -1)
    X_val_dct_vec   = X_val_dct.mean(axis=1).reshape(len(X_val_dct), -1)
    X_test_dct_vec  = X_test_dct.mean(axis=1).reshape(len(X_test_dct), -1)

    return (
        X_train_rgb, X_train_img_center, X_train_dct_vec, y_train,
        X_val_rgb,   X_val_img_center,   X_val_dct_vec,   y_val,
        X_test_rgb,  X_test_img_center,  X_test_dct_vec,  y_test,
    )


def make_tf_dataset(X_img, X_dct_vec, y, img_size: int, batch_size: int, training: bool):
    """
    Training dataset (single frame per video).
    EfficientNet expects preprocess_input on 0..255.
    Phase 1 saved RGB in 0..1.
    """
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    ds = tf.data.Dataset.from_tensor_slices((X_img, X_dct_vec, y))

    def _map(rgb, dct, label):
        rgb = tf.cast(rgb, tf.float32)  # 0..1
        rgb = tf.image.resize(rgb, (img_size, img_size))
        if training:
            rgb = tf.image.random_flip_left_right(rgb)
            rgb = tf.image.random_brightness(rgb, max_delta=0.1)

        rgb = preprocess_input(rgb * 255.0)

        dct = tf.cast(dct, tf.float32)
        label = tf.cast(label, tf.float32)
        return (rgb, dct), label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def predict_probs_single_frame(model, X_img, X_dct_vec, img_size: int, batch_size: int):
    """
    Predict probabilities for a single image array (N,H,W,3).
    IMPORTANT: dataset must yield ONE 'x' object (dict), not (rgb, dct) as a 2-tuple,
    otherwise Keras treats it as (x, y).
    """
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    ds = tf.data.Dataset.from_tensor_slices((X_img, X_dct_vec))

    def _map(rgb, dct):
        rgb = tf.cast(rgb, tf.float32)
        rgb = tf.image.resize(rgb, (img_size, img_size))
        rgb = preprocess_input(rgb * 255.0)
        dct = tf.cast(dct, tf.float32)

        # return ONE object containing both inputs
        return {"rgb_input": rgb, "dct_input": dct}

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    probs = model.predict(ds, verbose=0).squeeze()
    return np.array(probs)


def predict_probs_frame_averaging(model, X_rgb_seq, X_dct_vec, img_size: int, batch_size: int, frame_indices):
    """
    Predict by averaging probabilities across multiple frames per video.
    X_rgb_seq: (N, T, H, W, 3)
    frame_indices: list of indices in [0..T-1]
    """
    probs_all = []
    for idx in frame_indices:
        X_img = X_rgb_seq[:, idx]
        probs = predict_probs_single_frame(model, X_img, X_dct_vec, img_size, batch_size)
        probs_all.append(probs)
    probs_all = np.stack(probs_all, axis=0)          # (K, N)
    return probs_all.mean(axis=0)                    # (N,)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "balanced_acc",
    max_fpr: float | None = None,
):
    """
    Choose decision threshold τ on validation set.

    objective:
      - "balanced_acc"  : maximize (TPR + TNR)/2  [recommended]
      - "f1"            : maximize F1
      - "fake_recall"   : maximize recall for FAKE class (label=1)
      - "min_fnr"       : minimize FNR for FAKE class (i.e., minimize missed fakes)

    max_fpr:
      Optional constraint on false positive rate for FAKE predictions on REAL samples.
      Example: max_fpr=0.30 means "allow at most 30% of REAL videos to be flagged FAKE".
    """
    y_true = y_true.astype(int)

    # counts for REAL(0) and FAKE(1)
    n_real = int((y_true == 0).sum())
    n_fake = int((y_true == 1).sum())

    rows = []
    for tau in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= tau).astype(int)

        # confusion components
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        acc = (tp + tn) / max(n_real + n_fake, 1)

        # REAL recall = TNR (how many reals correctly kept real)
        real_recall = tn / max(n_real, 1)

        # FAKE recall = TPR (how many fakes caught)
        fake_recall = tp / max(n_fake, 1)

        # false positive rate = FP / (FP + TN) over REAL
        fpr = fp / max(n_real, 1)

        # false negative rate = FN / (FN + TP) over FAKE
        fnr = fn / max(n_fake, 1)

        bal_acc = 0.5 * (real_recall + fake_recall)

        # F1 for FAKE class (positive class = 1)
        # precision_fake = tp / (tp + fp)
        precision_fake = tp / max(tp + fp, 1)
        f1 = (2 * precision_fake * fake_recall) / max(precision_fake + fake_recall, 1e-12)

        rows.append({
            "tau": float(tau),
            "acc": acc,
            "bal_acc": bal_acc,
            "f1": f1,
            "real_recall": real_recall,
            "fake_recall": fake_recall,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    # optional constraint
    if max_fpr is not None:
        rows = [r for r in rows if r["fpr"] <= max_fpr]
        if not rows:
            raise ValueError(f"No threshold satisfied max_fpr={max_fpr}. Try a larger value.")

    if objective == "balanced_acc":
        key = lambda r: (r["bal_acc"], r["f1"])
    elif objective == "f1":
        key = lambda r: (r["f1"], r["bal_acc"])
    elif objective == "fake_recall":
        key = lambda r: (r["fake_recall"], r["bal_acc"])
    elif objective == "min_fnr":
        key = lambda r: (-r["fnr"], r["bal_acc"])  # maximize -fnr == minimize fnr
    else:
        raise ValueError("objective must be one of: balanced_acc, f1, fake_recall, min_fnr")

    best = max(rows, key=key)

    # Print top 10 for visibility
    top = sorted(rows, key=key, reverse=True)[:10]
    print("\nTop thresholds on validation:")
    print(" tau |  acc | bal_acc |   f1 | real_rec | fake_rec |  fpr |  fnr")
    for r in top:
        print(f"{r['tau']:.2f} | {r['acc']:.3f} |  {r['bal_acc']:.3f} | {r['f1']:.3f} |"
              f"   {r['real_recall']:.3f} |   {r['fake_recall']:.3f} | {r['fpr']:.3f} | {r['fnr']:.3f}")

    return best["tau"], best



def main():
    # (optional) reproducibility
    SEED = 42
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ffpp", choices=["ffpp", "celebdf"])
    parser.add_argument("--processed_dir", type=str, default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fine_tune_layers", type=int, default=20)

    # NEW: frame averaging controls
    parser.add_argument("--avg_frames", type=int, default=3, choices=[1, 3, 5],
                        help="How many frames to average for val/test predictions (1=center only)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="If set, use this threshold; if not set, pick best τ from validation")
    parser.add_argument("--init_model_path", type=str, default=None,
                    help="Path to a .keras model to continue training from (Stage 2)")

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    npz_path = processed_dir / f"{args.dataset}_rgb_dct_splits.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")

    data = load_npz(npz_path)
    (
        X_train_rgb_seq, X_train_img_center, X_train_dct_vec, y_train,
        X_val_rgb_seq,   X_val_img_center,   X_val_dct_vec,   y_val,
        X_test_rgb_seq,  X_test_img_center,  X_test_dct_vec,  y_test,
    ) = make_inputs_from_npz(data)

    dct_dim = X_train_dct_vec.shape[1]
    T = X_train_rgb_seq.shape[1]
    print("RGB train seq:", X_train_rgb_seq.shape)
    print("RGB train center:", X_train_img_center.shape)
    print("DCT train:", X_train_dct_vec.shape)
    print("Labels:", y_train.shape, np.bincount(y_train.astype(int)))

    # Train/val/test datasets for training (center-frame)
    train_ds = make_tf_dataset(X_train_img_center, X_train_dct_vec, y_train,
                               img_size=args.img_size, batch_size=args.batch_size, training=True)
    val_ds   = make_tf_dataset(X_val_img_center, X_val_dct_vec, y_val,
                               img_size=args.img_size, batch_size=args.batch_size, training=False)
    test_ds  = make_tf_dataset(X_test_img_center, X_test_dct_vec, y_test,
                               img_size=args.img_size, batch_size=args.batch_size, training=False)

    if args.init_model_path:
        print(f"Loading model from: {args.init_model_path}")
        model = tf.keras.models.load_model(args.init_model_path)

        # Find backbone by name (efficientnetb0)
        backbone = model.get_layer("efficientnetb0")
        set_finetune_layers(backbone, args.fine_tune_layers)

        # IMPORTANT: recompile to apply new LR + updated trainable layers
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        model = build_hybrid_model(
            img_size=args.img_size,
            dct_dim=dct_dim,
            fine_tune_layers=args.fine_tune_layers,
            lr=args.lr,
        )

    model.summary()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = CHECKPOINT_DIR / f"{args.dataset}_best_{run_id}_ft{args.fine_tune_layers}_lr{args.lr}.keras"
    final_path = SAVED_DIR / f"{args.dataset}_final_{run_id}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\nEvaluating (center-frame) on test set...")
    test_results = model.evaluate(test_ds, return_dict=True, verbose=1)
    print("Test results (center-frame):", test_results)

    # ---- NEW: choose frame indices for averaging ----
    if args.avg_frames == 1:
        frame_indices = [T // 2]
    elif args.avg_frames == 3:
        frame_indices = [T // 4, T // 2, (3 * T) // 4]
    else:  # 5
        frame_indices = [T // 6, T // 3, T // 2, (2 * T) // 3, (5 * T) // 6]

    print(f"\nUsing frame averaging over indices: {frame_indices} (T={T})")

    # ---- NEW: threshold selection using validation set ----
    val_prob = predict_probs_frame_averaging(
        model, X_val_rgb_seq, X_val_dct_vec,
        img_size=args.img_size, batch_size=args.batch_size,
        frame_indices=frame_indices
    )

    if args.threshold is None:
        tau, stats = find_best_threshold(y_val, val_prob, objective="balanced_acc")
        print(f"Selected τ={tau:.2f} by balanced_acc (val bal_acc={stats['bal_acc']:.3f}, f1={stats['f1']:.3f})")
    else:
        tau = float(args.threshold)
        print(f"Using provided τ = {tau:.2f}")

    # ---- NEW: test evaluation using the same averaging + chosen τ ----
    test_prob = predict_probs_frame_averaging(
        model, X_test_rgb_seq, X_test_dct_vec,
        img_size=args.img_size, batch_size=args.batch_size,
        frame_indices=frame_indices
    )

    y_true = y_test.astype(int)
    y_pred = (test_prob >= tau).astype(int)

    print(f"\nFinal report (avg_frames={args.avg_frames}, τ={tau:.2f})")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0
    ))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # Save final model
    model.save(str(final_path))
    print("\nSaved best checkpoint to:", best_path)
    print("Saved final model to:", final_path)


if __name__ == "__main__":
    if __package__ is None:
        sys.path.append(str(REPO_ROOT))
    main()
