"""
Samples a handful of evenly-spaced JPEG frames from a video file.

We deliberately sample from the ANNOTATED output video (not the raw
upload) because pipeline.py already burns in team-colored boxes
(cyan = Team A, gold = Team B) and jersey numbers. That makes it much
easier for the vision model to tell the two sides apart and refer to
specific players, instead of guessing from a plain broadcast frame.
"""
from __future__ import annotations

import cv2


def sample_frames(video_path: str, max_frames: int = 8) -> list[bytes]:
    """Return up to `max_frames` JPEG-encoded frames, evenly spaced."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame sampling: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Fallback: some containers don't report frame count reliably.
            frames_bytes = []
            while len(frames_bytes) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    frames_bytes.append(buf.tobytes())
            return frames_bytes

        step = max(total // max_frames, 1)
        indices = list(range(0, total, step))[:max_frames]

        frames_bytes = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                frames_bytes.append(buf.tobytes())
        return frames_bytes
    finally:
        cap.release()
