import cv2
import numpy as np
from typing import Tuple, Optional
from .crop_faces import crop_face_or_fullframe


def extract_frames_from_video(
    video_path: str,
    output_size: Tuple[int, int] = (224, 224),
    frame_count: int = 15,
    padding_factor: float = 1.3,
) -> np.ndarray:
    """
    Sample `frame_count` frames uniformly across the video.
    For each frame: detect face -> crop -> resize.
    Returns shape: (frame_count, H, W, 3) or empty array if failed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.array([])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.array([])

    step = max(total_frames // frame_count, 1)

    frames = []
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        cropped = crop_face_or_fullframe(frame, padding_factor=padding_factor)
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
        frames.append(resized)

    cap.release()

    if len(frames) != frame_count:
        return np.array([])

    return np.array(frames, dtype=np.uint8)
