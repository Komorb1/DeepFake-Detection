import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from preprocessing.extract_frames import extract_frames_from_video

# repo root = DeepFake/
REPO_ROOT = Path(__file__).resolve().parents[1]

# FaceForensics++ folders (pipeline structure)
FF_REAL_DIR = REPO_ROOT / "FF" / "original"
FF_FAKE_DIR = REPO_ROOT / "FF" / "Deepfakes"

# Celeb-DF folders (pipeline structure)
CELEB_REAL_DIR = REPO_ROOT / "Celeb" / "Celeb-real"
CELEB_FAKE_DIR = REPO_ROOT / "Celeb" / "Celeb-fake"

# output (kept inside repo)
DEFAULT_OUT_DIR = REPO_ROOT / "processed"


def extract_dct_features(frames: np.ndarray, dct_size: int = 64) -> np.ndarray:
    """
    frames: (T, H, W, 3) uint8
    returns: (T, dct_size, dct_size) float32   (RAW DCT values)
    """
    import cv2

    dct_frames = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (dct_size, dct_size), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(gray))
        dct_frames.append(dct)
    return np.array(dct_frames, dtype=np.float32)


def one_hot(y: np.ndarray, num_classes: int = 2) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[y]


def list_videos(folder: Path, max_videos: int):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files[:max_videos] if max_videos > 0 else files


def normalize_dct_with_train(train_dct_raw, val_dct_raw, test_dct_raw, eps=1e-8):
    """
    Normalize DCT arrays using TRAIN RAW stats only (avoid leakage).

    Inputs:
      *_dct_raw: (N, T, dct, dct) float32
    Returns:
      train_norm, val_norm, test_norm, dmin, dmax
    """
    dmin = float(train_dct_raw.min())
    dmax = float(train_dct_raw.max())
    denom = (dmax - dmin) + eps
    train = (train_dct_raw - dmin) / denom
    val = (val_dct_raw - dmin) / denom
    test = (test_dct_raw - dmin) / denom
    return train, val, test, dmin, dmax


def print_dataset_info(X_train_rgb, X_train_dct, X_train_dct_raw, y_train):
    print("\n=== DATASET INFORMATION ===")
    print(f"RGB Train:      {X_train_rgb.shape} (float32, normalized 0..1)")
    print(f"DCT Train RAW:  {X_train_dct_raw.shape} (float32, unnormalized)")
    print(f"DCT Train NORM: {X_train_dct.shape} (float32, normalized using train raw min/max)")
    print(f"y_train:        {y_train.shape}")
    print(f"REAL: {int((y_train == 0).sum())} | FAKE: {int((y_train == 1).sum())}")


def process_one_dataset(
    dataset_name: str,
    real_dir: Path,
    fake_dir: Path,
    out_dir: Path,
    img_size: int,
    frame_count: int,
    dct_size: int,
    padding_factor: float,
    max_videos: int,
    seed: int,
):
    real_videos = list_videos(real_dir, max_videos)
    fake_videos = list_videos(fake_dir, max_videos)

    X_rgb, X_dct_raw, y, meta_rows = [], [], [], []

    print(f"\n=== [{dataset_name.upper()}] ===")
    print(f"REAL dir: {real_dir}")
    print(f"FAKE dir: {fake_dir}")
    print(f"Loading REAL videos: {len(real_videos)}")

    for vp in tqdm(real_videos, desc=f"{dataset_name}-REAL"):
        frames = extract_frames_from_video(
            str(vp),
            output_size=(img_size, img_size),
            frame_count=frame_count,
            padding_factor=padding_factor,
        )
        if frames.size == 0:
            continue

        dct = extract_dct_features(frames, dct_size=dct_size)

        X_rgb.append(frames)
        X_dct_raw.append(dct)
        y.append(0)
        meta_rows.append({"path": str(vp), "label": 0, "dataset": dataset_name})

    print(f"Loading FAKE videos: {len(fake_videos)}")
    for vp in tqdm(fake_videos, desc=f"{dataset_name}-FAKE"):
        frames = extract_frames_from_video(
            str(vp),
            output_size=(img_size, img_size),
            frame_count=frame_count,
            padding_factor=padding_factor,
        )
        if frames.size == 0:
            continue

        dct = extract_dct_features(frames, dct_size=dct_size)

        X_rgb.append(frames)
        X_dct_raw.append(dct)
        y.append(1)
        meta_rows.append({"path": str(vp), "label": 1, "dataset": dataset_name})

    if len(X_rgb) == 0:
        raise RuntimeError(f"[{dataset_name}] No videos were successfully processed.")

    # Arrays (still pre-split)
    X_rgb = np.array(X_rgb, dtype=np.uint8)                 # (N, T, H, W, 3)
    X_dct_raw = np.array(X_dct_raw, dtype=np.float32)       # (N, T, dct, dct)  RAW
    y = np.array(y, dtype=np.int64)

    # Split: 70% train, 15% val, 15% test  (use RAW DCT for splits!)
    X_temp_rgb, X_test_rgb, X_temp_dct_raw, X_test_dct_raw, y_temp, y_test = train_test_split(
        X_rgb, X_dct_raw, y, test_size=0.15, random_state=seed, stratify=y
    )
    val_frac_of_temp = 0.15 / 0.85
    X_train_rgb, X_val_rgb, X_train_dct_raw, X_val_dct_raw, y_train, y_val = train_test_split(
        X_temp_rgb, X_temp_dct_raw, y_temp, test_size=val_frac_of_temp, random_state=seed, stratify=y_temp
    )

    # Normalize RGB to 0..1
    X_train_rgb = X_train_rgb.astype(np.float32) / 255.0
    X_val_rgb   = X_val_rgb.astype(np.float32) / 255.0
    X_test_rgb  = X_test_rgb.astype(np.float32) / 255.0

    # Normalize DCT using TRAIN RAW stats only
    X_train_dct, X_val_dct, X_test_dct, dmin, dmax = normalize_dct_with_train(
        X_train_dct_raw, X_val_dct_raw, X_test_dct_raw
    )

    y_train_cat = one_hot(y_train, 2)
    y_val_cat   = one_hot(y_val, 2)
    y_test_cat  = one_hot(y_test, 2)

    print_dataset_info(X_train_rgb, X_train_dct, X_train_dct_raw, y_train)
    print(f"[{dataset_name}] Done. Train={len(X_train_rgb)} Val={len(X_val_rgb)} Test={len(X_test_rgb)}")
    print(f"[{dataset_name}] DCT train min/max (RAW): {dmin:.6f} / {dmax:.6f}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{dataset_name}_rgb_dct_splits.npz"
    np.savez_compressed(
        npz_path,
        # RGB splits (normalized 0..1)
        X_train_rgb=X_train_rgb, X_val_rgb=X_val_rgb, X_test_rgb=X_test_rgb,

        # DCT splits (normalized)  [backward compatible with train.py]
        X_train_dct=X_train_dct, X_val_dct=X_val_dct, X_test_dct=X_test_dct,

        # New: RAW DCT splits (needed for fair cross-dataset eval)
        X_train_dct_raw=X_train_dct_raw, X_val_dct_raw=X_val_dct_raw, X_test_dct_raw=X_test_dct_raw,

        # labels
        y_train=y_train, y_val=y_val, y_test=y_test,
        y_train_cat=y_train_cat, y_val_cat=y_val_cat, y_test_cat=y_test_cat,

        # stats used for normalization (from TRAIN RAW)
        dct_train_min=dmin, dct_train_max=dmax,
    )

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = out_dir / f"{dataset_name}_processed_videos.csv"
    meta_df.to_csv(meta_csv, index=False)

    print(f"[{dataset_name}] Saved: {npz_path}")
    print(f"[{dataset_name}] Saved: {meta_csv}")


def main():
    parser = argparse.ArgumentParser(description="Prepare RGB+DCT datasets for FF and Celeb in one run.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Output folder to save .npz and metadata")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--frame_count", type=int, default=15)
    parser.add_argument("--dct_size", type=int, default=64)
    parser.add_argument("--padding_factor", type=float, default=1.3)
    parser.add_argument("--max_videos", type=int, default=200, help="Max videos per class (set 0 for all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Safety checks
    for p in [FF_REAL_DIR, FF_FAKE_DIR, CELEB_REAL_DIR, CELEB_FAKE_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    # Run FFPP
    process_one_dataset(
        dataset_name="ffpp",
        real_dir=FF_REAL_DIR,
        fake_dir=FF_FAKE_DIR,
        out_dir=out_dir,
        img_size=args.img_size,
        frame_count=args.frame_count,
        dct_size=args.dct_size,
        padding_factor=args.padding_factor,
        max_videos=args.max_videos,
        seed=args.seed,
    )

    # Run CelebDF
    process_one_dataset(
        dataset_name="celebdf",
        real_dir=CELEB_REAL_DIR,
        fake_dir=CELEB_FAKE_DIR,
        out_dir=out_dir,
        img_size=args.img_size,
        frame_count=args.frame_count,
        dct_size=args.dct_size,
        padding_factor=args.padding_factor,
        max_videos=args.max_videos,
        seed=args.seed,
    )

    print("\nAll preprocessing finished successfully!")


if __name__ == "__main__":
    # Allows running both:
    # 1) python -m preprocessing.prepare_dataset
    # 2) python preprocessing/prepare_dataset.py
    if __package__ is None:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
    main()
