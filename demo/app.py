# demo/app.py
# Deepfake detection demo (image + optional video) using hybrid model + Grad-CAM

from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Ensure repo root is importable (so: import xai.gradCam works)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import IMG_SIZE, TAU_DEFAULT  # noqa: E402
from model import load_model, prepare_inputs_from_pil, predict_prob_fake  # noqa: E402
from xai.gradCam import compute_gradcam, overlay_heatmap  # noqa: E402


@st.cache_resource
def _get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face_opencv(pil_img: Image.Image, padding: float = 0.35) -> Image.Image:
    """Largest-face crop with padding. Falls back to center-crop if no face found."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = _get_face_cascade()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    h, w = img.shape[:2]
    if len(faces) == 0:
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = img[y0 : y0 + s, x0 : x0 + s]
        return Image.fromarray(crop)

    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])

    px = int(fw * padding)
    py = int(fh * padding)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w, x + fw + px)
    y1 = min(h, y + fh + py)

    crop = img[y0:y1, x0:x1]
    return Image.fromarray(crop)


@st.cache_resource
def get_model():
    return load_model()


def _safe_prepare(face_img: Image.Image):
    """
    Supports BOTH versions of prepare_inputs_from_pil:
      A) returns inputs dict
      B) returns (inputs, bgr_01, dct_vec)
    """
    out = prepare_inputs_from_pil(face_img)

    # B) preferred
    if isinstance(out, (tuple, list)) and len(out) == 3:
        inputs, bgr_01, dct_vec = out
        bgr_01 = np.asarray(bgr_01, dtype=np.float32)
        dct_vec = np.asarray(dct_vec, dtype=np.float32).reshape(-1)
        return inputs, bgr_01, dct_vec

    # A) fallback: derive what we can for Grad-CAM
    inputs = out
    rgb = np.array(face_img.convert("RGB"), dtype=np.uint8)
    bgr_01 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

    # If dct_input exists, use it (shape: (1,4096) or (4096,))
    dct_vec = None
    if isinstance(inputs, dict) and "dct_input" in inputs:
        d = inputs["dct_input"]
        d = np.array(d)
        dct_vec = d.reshape(-1).astype(np.float32)
    else:
        dct_vec = np.zeros((4096,), dtype=np.float32)

    return inputs, bgr_01, dct_vec


def _label_from_prob(prob_fake: float, tau: float, margin: float) -> tuple[str, float]:
    """Returns (label_string, confidence)."""
    if abs(prob_fake - tau) < margin:
        label = "UNCERTAIN (leaning FAKE)" if prob_fake >= tau else "UNCERTAIN (leaning REAL)"
    else:
        label = "FAKE" if prob_fake >= tau else "REAL"

    if label.startswith("FAKE"):
        conf = prob_fake
    elif label.startswith("REAL"):
        conf = 1.0 - prob_fake
    else:
        conf = max(0.0, 1.0 - (abs(prob_fake - tau) / max(margin, 1e-6)))

    return label, float(conf)


def _sample_video_frames(video_path: str, num_frames: int = 12) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        # fallback: just read sequentially
        idxs = list(range(num_frames))
    else:
        idxs = np.linspace(0, max(frame_count - 1, 0), num=num_frames).astype(int).tolist()

    frames: list[Image.Image] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def main():
    st.set_page_config(page_title="Deepfake Detection Demo", layout="centered")
    st.title("Deepfake Detection Demo")
    st.write("Upload an image (or a short video) to see REAL/FAKE prediction and a Grad-CAM heatmap.")

    with st.sidebar:
        st.header("Settings")
        mode = st.selectbox("Input type", ["Image", "Video"], index=0)
        tau = st.slider("Decision threshold (τ)", 0.0, 1.0, float(TAU_DEFAULT), 0.01)
        margin = st.slider("Uncertainty margin (around τ)", 0.0, 0.20, 0.05, 0.01)
        alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
        padding = st.slider("Face crop padding", 0.0, 0.80, 0.35, 0.05)
        use_crop = st.checkbox("Crop face (recommended)", value=True)

    model = get_model()

    if mode == "Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.info("Upload a JPG/PNG image to run inference.")
            return

        pil_img = Image.open(uploaded).convert("RGB")
        face_img = crop_face_opencv(pil_img, padding=padding) if use_crop else pil_img

        c1, c2 = st.columns(2)
        with c1:
            st.image(pil_img, caption="Original", use_container_width=True)
        with c2:
            st.image(face_img, caption="Face crop (used)", use_container_width=True)

        inputs, bgr_01, dct_vec = _safe_prepare(face_img)

        prob_fake = float(predict_prob_fake(model, inputs))
        pred_label, confidence = _label_from_prob(prob_fake, tau, margin)

        st.markdown(
            f"**Prediction:** {pred_label}  "
            f"(prob_fake={prob_fake:.3f}, confidence={confidence:.3f}, τ={tau:.2f})"
        )

        # Grad-CAM target = predicted side of tau
        target = "fake" if prob_fake >= tau else "real"
        try:
            heat = compute_gradcam(
                model=model,
                img_01=bgr_01,
                dct_vec=dct_vec,
                target_layer=None,
                img_size=IMG_SIZE,
                target=target,
            )

            # ensure image matches heatmap size before overlay
            bgr_cam = bgr_01
            if bgr_cam.ndim == 4:
                bgr_cam = bgr_cam[0]
            bgr_cam = np.clip(bgr_cam, 0.0, 1.0).astype(np.float32)
            bgr_cam = cv2.resize(bgr_cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            overlay_bgr = overlay_heatmap(bgr_cam, heat, alpha=float(alpha))
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


            st.image(overlay_rgb, caption="Grad-CAM overlay", use_container_width=True)
            st.write("heatmap stats:", float(heat.min()), float(heat.max()), float(heat.mean()))

            if float(heat.max()) < 1e-6:
                st.warning("Grad-CAM produced an empty heatmap for this input (model may be uncertain).")
        except Exception as e:
            st.warning(f"Grad-CAM failed for this input: {e}")

        return

    # ---------------- VIDEO MODE ----------------
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is None:
        st.info("Upload a video to run inference.")
        return

    # Save to a temp file (OpenCV VideoCapture needs a path)
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    frames = _sample_video_frames(video_path, num_frames=12)
    if not frames:
        st.error("Could not read frames from this video.")
        return

    probs = []
    show_frames = []
    for f in frames:
        face = crop_face_opencv(f, padding=padding) if use_crop else f
        inputs, _, _ = _safe_prepare(face)
        p = float(predict_prob_fake(model, inputs))
        probs.append(p)
        show_frames.append(face)

    probs = np.array(probs, dtype=np.float32)
    prob_fake_avg = float(probs.mean())

    pred_label, confidence = _label_from_prob(prob_fake_avg, tau, margin)
    st.markdown(
        f"**Video prediction (avg over {len(probs)} frames):** {pred_label}  "
        f"(avg_prob_fake={prob_fake_avg:.3f}, confidence={confidence:.3f}, τ={tau:.2f})"
    )

    st.line_chart(probs)

    # Show Grad-CAM for the middle sampled frame
    mid = len(show_frames) // 2
    face_img = show_frames[mid]
    st.image(face_img, caption="Frame used for Grad-CAM", use_container_width=True)

    inputs, bgr_01, dct_vec = _safe_prepare(face_img)
    target = "fake" if prob_fake_avg >= tau else "real"
    try:
        heat = compute_gradcam(
            model=model,
            img_01=bgr_01,
            dct_vec=dct_vec,
            target_layer=None,
            img_size=IMG_SIZE,
            target=target,
        )

        # ensure image matches heatmap size before overlay
        bgr_cam = bgr_01
        if bgr_cam.ndim == 4:
            bgr_cam = bgr_cam[0]
        bgr_cam = np.clip(bgr_cam, 0.0, 1.0).astype(np.float32)
        bgr_cam = cv2.resize(bgr_cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        overlay_bgr = overlay_heatmap(bgr_cam, heat, alpha=float(alpha))
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        st.image(overlay_rgb, caption="Grad-CAM overlay (one frame)", use_container_width=True)
        st.write("heatmap stats:", float(heat.min()), float(heat.max()), float(heat.mean()))
    except Exception as e:
        st.warning(f"Grad-CAM failed on video frame: {e}")


if __name__ == "__main__":
    main()
