import cv2
import dlib
import numpy as np
from typing import Optional, Tuple

# Dlib frontal face detector (no shape predictor needed)
_FACE_DETECTOR = dlib.get_frontal_face_detector()


def _largest_face(rects):
    # Pick the largest face (more stable than "faces[0]")
    best = None
    best_area = -1
    for r in rects:
        area = max(0, (r.right() - r.left())) * max(0, (r.bottom() - r.top()))
        if area > best_area:
            best_area = area
            best = r
    return best


def crop_face_or_fullframe(
    frame_bgr: np.ndarray,
    padding_factor: float = 1.3,
) -> np.ndarray:
    """
    Detect face and crop with padding. If no face is found, returns full frame.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rects = _FACE_DETECTOR(gray)

    h, w = frame_bgr.shape[:2]
    if len(rects) == 0:
        return frame_bgr

    r = _largest_face(rects)

    y_center = (r.top() + r.bottom()) // 2
    x_center = (r.left() + r.right()) // 2
    rh = r.bottom() - r.top()
    rw = r.right() - r.left()

    h_pad = int(rh * padding_factor)
    w_pad = int(rw * padding_factor)

    y1 = max(0, y_center - h_pad // 2)
    y2 = min(h, y_center + h_pad // 2)
    x1 = max(0, x_center - w_pad // 2)
    x2 = min(w, x_center + w_pad // 2)

    cropped = frame_bgr[y1:y2, x1:x2]
    return cropped if cropped.size != 0 else frame_bgr
