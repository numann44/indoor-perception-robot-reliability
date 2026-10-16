"""Seeded sensor corruptions. Evaluation labels are never accepted or altered."""

from dataclasses import replace
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter


CONDITIONS = ["clean"] + [f"{name}:{level}" for name in
    ("jpeg", "blur", "low_light", "frame_drop", "timestamp_jitter", "depth_dropout")
    for level in (1, 2, 3)]
SEVERITIES = {"jpeg_quality": [70, 40, 15], "blur_sigma_pixels": [0.75, 1.5, 3.0],
              "low_light_rgb_multiplier": [0.7, 0.4, 0.15], "frame_drop_probability": [0.05, 0.15, 0.30],
              "timestamp_jitter_uniform_bound_ms": [5, 20, 50], "full_depth_dropout_probability": [0.10, 0.30, 0.60]}


def corrupt_frame(frame, condition: str, rng: np.random.Generator):
    """Return a new sensor frame or None for a dropped frame.

    Existing RGB/depth arrays are not mutated. Timestamp jitter models sensor clock
    jitter, not pixel motion. Dropped frames are counted separately in evaluation.
    """
    if condition == "clean":
        return frame
    try:
        name, raw_level = condition.split(":")
        level = int(raw_level)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid corruption condition: {condition}") from exc
    if level not in (1, 2, 3) or condition not in CONDITIONS:
        raise ValueError(f"Unknown corruption condition: {condition}")
    i = level - 1
    rgb = frame.rgb
    if name == "jpeg":
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=(70, 40, 15)[i])
        buffer.seek(0)
        with Image.open(buffer) as img:
            rgb = np.array(img.convert("RGB"))
    elif name == "blur":
        rgb = np.array(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur((0.75, 1.5, 3.0)[i])))
    elif name == "low_light":
        rgb = np.rint(rgb.astype(np.float32) * (0.7, 0.4, 0.15)[i]).clip(0, 255).astype(np.uint8)
    elif name == "frame_drop":
        if rng.random() < (0.05, 0.15, 0.30)[i]:
            return None
    elif name == "timestamp_jitter":
        bound = (5_000_000, 20_000_000, 50_000_000)[i]
        return replace(frame, timestamp_ns=max(0, frame.timestamp_ns + int(rng.integers(-bound, bound + 1))))
    elif name == "depth_dropout":
        if rng.random() < (0.10, 0.30, 0.60)[i]:
            return replace(frame, depth=None)
    return replace(frame, rgb=rgb)
