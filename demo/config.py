from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "model" / "checkpoints" / "ffpp_best_20251214_000015_ft80_lr5e-06.keras"
DCT_STATS_PATH = ROOT / "processed" / "ffpp_rgb_dct_splits.npz"

IMG_SIZE = 224
DCT_SIZE = 64
TAU_DEFAULT = 0.55
