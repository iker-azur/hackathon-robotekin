"""Vision configuration: built-in defaults merged with config/vision.json."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "vision.json")
)

# Color ranges in OpenCV HSV units: H 0-179, S 0-255, V 0-255.
# Each entry is [h_min, h_max, s_min, s_max, v_min, v_max].
DEFAULT_COLOR_RANGES: Dict[str, list] = {
    "rojo": [[0, 8, 90, 255, 60, 255], [170, 179, 90, 255, 60, 255]],
    "naranja": [[9, 20, 120, 255, 80, 255]],
    "amarillo": [[21, 34, 90, 255, 80, 255]],
    "verde": [[35, 85, 70, 255, 40, 255]],
    "azul": [[86, 130, 70, 255, 40, 255]],
    "morado": [[131, 160, 50, 255, 40, 255]],
    "blanco": [[0, 179, 0, 40, 190, 255]],
    "gris": [[0, 179, 0, 40, 60, 189]],
    "negro": [[0, 179, 0, 255, 0, 50]],
}

DEFAULTS: Dict[str, Any] = {
    # "sim:demo" runs without hardware; use 0 (or 1, ...) for a real webcam.
    "camera": {"index": "sim:demo", "width": 640, "height": 480, "display_fps": 12},
    "roi": None,  # [x, y, w, h] in camera pixels, or null for the full frame
    "foreground": {
        "min_saturation": 70,
        "min_value": 60,
        # Dark (black) and bright achromatic (white) objects are opt-in: they need a
        # background of the opposite tone, otherwise the whole table is segmented.
        "detect_dark": False,
        "dark_value_max": 50,
        "detect_bright_achromatic": False,
        "bright_value_min": 190,
        "achromatic_saturation_max": 40,
    },
    "morphology": {"kernel_size": 5, "open_iterations": 1, "close_iterations": 2},
    "shape": {
        "min_area_px": 400,
        "max_area_px": 60000,
        "approx_epsilon_ratio": 0.03,
        "circularity_min": 0.85,
        "square_aspect_tolerance": 0.15,
    },
    "color": {"ranges": DEFAULT_COLOR_RANGES, "min_pixel_fraction": 0.6},
    "tracking": {
        "max_distance_px": 60,
        "max_missed_frames": 8,
        "motion_threshold_px": 6,
        "min_hits": 2,
        "history_length": 7,
    },
    "quality": {
        "min_solidity": 0.88,  # below this, corners are missing or the outline is chipped
        "expected_colors": None,  # e.g. ["rojo", "azul"]; null accepts any color
        "expected_types": None,  # e.g. ["cubo"]; null accepts any type
        "expected_area_px": None,  # [min, max] accepted area; null disables the check
        "expected_area_tolerance": 0.35,
        "min_color_confidence": 0.6,
    },
    "calibration_path": None,
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the effective configuration. A missing file means defaults only."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as f:
        return _deep_merge(DEFAULTS, json.load(f))


def save_roi(path: str, roi: Optional[list]) -> None:
    """Persist the workspace ROI, keeping every other key of the config file intact."""
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data["roi"] = list(roi) if roi else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
