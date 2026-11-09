"""Passive detection of the opponent-search Next button in a screenshot."""

import argparse
import json
import math
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "next_button_config.json"


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return cv2


def load_config(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read Next-button config: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Next-button config must be a JSON object")
    template = config.get("template")
    if not isinstance(template, str) or not template:
        raise ValueError("Next-button template path must be a non-empty string")
    threshold = config.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Next-button threshold must be between 0 and 1")
    scales = config.get("scales")
    if (not isinstance(scales, list) or not scales
            or any(isinstance(scale, bool)
                   or not isinstance(scale, (int, float))
                   or not math.isfinite(scale) or scale <= 0
                   for scale in scales)):
        raise ValueError("Next-button scales must be positive numbers")
    region = config.get("search_region")
    if not isinstance(region, dict) or set(region) != {
        "x", "y", "width", "height"
    }:
        raise ValueError("Search region must define x, y, width, and height")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in region.values()):
        raise ValueError("Search region values must be finite numbers")
    if (region["x"] < 0 or region["y"] < 0 or region["width"] <= 0
            or region["height"] <= 0 or region["x"] + region["width"] > 1
            or region["y"] + region["height"] > 1):
        raise ValueError("Search region must fit within normalized image bounds")
    config["resolved_template"] = config_path.parent / template
    return config


def _match(screen, template, scales):
    cv2 = _opencv()
    screen_height, screen_width = screen.shape[:2]
    template_height, template_width = template.shape[:2]
    best = None
    for scale in scales:
        width = max(1, int(round(template_width * scale)))
        height = max(1, int(round(template_height * scale)))
        if width > screen_width or height > screen_height:
            continue
        resized = cv2.resize(
            template, (width, height), interpolation=cv2.INTER_AREA
        )
        result = cv2.matchTemplate(
            screen, resized, cv2.TM_CCOEFF_NORMED
        )
        _, score, _, location = cv2.minMaxLoc(result)
        if not math.isfinite(score):
            continue
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "scale": float(scale),
                "x": int(location[0]),
                "y": int(location[1]),
                "width": width,
                "height": height,
            }
    return best


def recognize_next_button(image_path, config_path=DEFAULT_CONFIG):
    """Detect the Next button in a static image; never clicks or interacts."""
    cv2 = _opencv()
    image_path = Path(image_path)
    config = load_config(config_path)
    result = {
        "image": str(image_path),
        "status": "not_detected",
        "detected": False,
        "confidence": None,
        "threshold": float(config["threshold"]),
        "scale": None,
        "box": None,
        "error": None,
    }
    if not image_path.is_file():
        result["status"] = "invalid_screenshot"
        result["error"] = f"Screenshot file not found: {image_path}"
        return result
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        result["status"] = "unreadable_screenshot"
        result["error"] = f"Could not decode screenshot: {image_path}"
        return result
    template_path = config["resolved_template"]
    if not template_path.is_file():
        result["status"] = "missing_template"
        result["error"] = f"Next-button template not found: {template_path}"
        return result
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None or template.size == 0:
        result["status"] = "invalid_template"
        result["error"] = f"Could not decode template: {template_path}"
        return result

    height, width = image.shape[:2]
    region = config["search_region"]
    left = max(0, min(width - 1, int(round(region["x"] * width))))
    top = max(0, min(height - 1, int(round(region["y"] * height))))
    right = max(left + 1, min(
        width, int(round((region["x"] + region["width"]) * width))
    ))
    bottom = max(top + 1, min(
        height, int(round((region["y"] + region["height"]) * height))
    ))
    match = _match(
        image[top:bottom, left:right], template, config["scales"]
    )
    if match is None:
        result["status"] = "no_match_region"
        return result

    result["confidence"] = match["score"]
    result["scale"] = match["scale"]
    result["detected"] = match["score"] >= config["threshold"]
    result["status"] = "detected" if result["detected"] else "not_detected"
    result["box"] = {
        "x": left + match["x"],
        "y": top + match["y"],
        "width": match["width"],
        "height": match["height"],
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Passively detect the Next button in a static screenshot."
    )
    parser.add_argument("image", help="path to a static screenshot")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Next-button config (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        result = recognize_next_button(args.image, args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Recognition error: {exc}", file=sys.stderr)
        return 2
    if result["confidence"] is None:
        print(f"NEXT BUTTON: {result['status'].upper()}")
    else:
        print(
            f"NEXT BUTTON: {'DETECTED' if result['detected'] else 'NOT DETECTED'} "
            f"(confidence={result['confidence']:.3f}, "
            f"threshold={result['threshold']:.2f}, scale={result['scale']:.2f})"
        )
    if result["error"]:
        print(f"WARNING: {result['error']}")
    return 0 if result["status"] not in (
        "invalid_screenshot", "unreadable_screenshot", "missing_template",
        "invalid_template",
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
