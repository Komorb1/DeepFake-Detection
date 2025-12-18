from typing import Dict
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from config import MODEL_PATH, IMG_SIZE, DCT_SIZE, DCT_STATS_PATH

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

_DCT_MINMAX = None  # cached


def _load_dct_minmax():
    global _DCT_MINMAX
    if _DCT_MINMAX is not None:
        return _DCT_MINMAX

    if not DCT_STATS_PATH.is_file():
        raise FileNotFoundError(
            f"DCT stats file not found: {DCT_STATS_PATH.resolve()}\n"
            "Set DCT_STATS_PATH in streamlit/config.py to your ffpp_rgb_dct_splits.npz"
        )

    stats = np.load(str(DCT_STATS_PATH), allow_pickle=True)
    dmin = float(stats["dct_train_min"])
    dmax = float(stats["dct_train_max"])
    _DCT_MINMAX = (dmin, dmax)
    return _DCT_MINMAX


def load_model() -> keras.Model:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH.resolve()}.\n"
            "Update MODEL_PATH in streamlit/config.py"
        )
    return keras.models.load_model(str(MODEL_PATH), compile=False)


def _pil_to_bgr_255(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype("float32")  # RGB 0..255
    bgr = rgb[..., ::-1]  # BGR 0..255
    return bgr


def _extract_dct_vector_from_bgr(bgr_255: np.ndarray) -> np.ndarray:
    dmin, dmax = _load_dct_minmax()

    gray = cv2.cvtColor(bgr_255.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype("float32")
    gray = cv2.resize(gray, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA)

    dct = cv2.dct(gray).astype("float32")  # RAW
    dct = (dct - dmin) / ((dmax - dmin) + 1e-8)
    dct = np.clip(dct, 0.0, 1.0)

    return dct.reshape(-1).astype("float32")  # (4096,)


def prepare_inputs_from_pil(pil_img: Image.Image) -> Dict[str, np.ndarray]:
    bgr_255 = _pil_to_bgr_255(pil_img)

    # EfficientNet preprocess (matches your training: preprocess_input(x*255))
    rgb_pre = preprocess_input(bgr_255)  # NOTE: we intentionally feed BGR to match training

    dct_vec = _extract_dct_vector_from_bgr(bgr_255)

    return {
        "rgb_input": np.expand_dims(rgb_pre.astype("float32"), axis=0),   # (1,224,224,3)
        "dct_input": np.expand_dims(dct_vec, axis=0),                    # (1,4096)
    }


def predict_prob_fake(model: keras.Model, inputs: Dict[str, np.ndarray]) -> float:
    pred = model.predict(inputs, verbose=0)
    return float(np.squeeze(pred))
