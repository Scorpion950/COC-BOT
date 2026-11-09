"""Passive template-based recognition for the configured Clash of Clans army."""

import argparse
import json
import math
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "army_recognition_config.json"


def load_config(config_path=DEFAULT_CONFIG):
    """Load and validate recognition labels, thresholds, and scales."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    templates = config.get("templates")
    scales = config.get("scales")
    if not isinstance(templates, list) or not templates:
        raise ValueError("config must define a non-empty 'templates' list")
    if not isinstance(scales, list) or not scales:
        raise ValueError("config must define a non-empty 'scales' list")
    if any(not isinstance(scale, (int, float)) or scale <= 0 for scale in scales):
        raise ValueError("every configured scale must be a positive number")

    seen_labels = set()
    for template in templates:
        if not isinstance(template, dict):
            raise ValueError("each template entry must be an object")
        label = template.get("label")
        path = template.get("path")
        threshold = template.get("threshold")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("every template must have a non-empty label")
        if label in seen_labels:
            raise ValueError(f"duplicate template label: {label}")
        if not isinstance(path, str) or not path:
            raise ValueError(f"template '{label}' must have a path")
        if (not isinstance(threshold, (int, float))
                or not 0 <= threshold <= 1):
            raise ValueError(
                f"template '{label}' threshold must be between 0 and 1"
            )
        seen_labels.add(label)
        template["resolved_path"] = config_path.parent / path

    return config


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for army recognition. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return cv2


def _read_image(path, description):
    cv2 = _opencv()
    if not Path(path).is_file():
        raise ValueError(f"{description} is missing or is not a readable image: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{description} is missing or is not a readable image: {path}")
    return image


def _match_template(screen, template, scales):
    cv2 = _opencv()
    screen_height, screen_width = screen.shape[:2]
    template_height, template_width = template.shape[:2]
    best = None

    for scale in scales:
        width = max(1, int(round(template_width * scale)))
        height = max(1, int(round(template_height * scale)))
        if width > screen_width or height > screen_height:
            continue
        resized = cv2.resize(template, (width, height),
                             interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen, resized, cv2.TM_CCOEFF_NORMED)
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


def _intersection_over_union(left, right):
    x1 = max(left["x"], right["x"])
    y1 = max(left["y"], right["y"])
    x2 = min(left["x"] + left["width"], right["x"] + right["width"])
    y2 = min(left["y"] + left["height"], right["y"] + right["height"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = left["width"] * left["height"]
    right_area = right["width"] * right["height"]
    return intersection / float(left_area + right_area - intersection)


def recognize_image(image_path, config_path=DEFAULT_CONFIG):
    """Recognize configured army items in a static image without interacting."""
    config = load_config(config_path)
    image_path = Path(image_path)
    screen = _read_image(image_path, "Screenshot")
    detections = []

    for item in config["templates"]:
        detection = {
            "label": item["label"],
            "threshold": float(item["threshold"]),
            "status": "missing_template",
            "score": None,
            "scale": None,
            "box": None,
        }
        template_path = item["resolved_path"]
        if not template_path.is_file():
            detections.append(detection)
            continue

        try:
            template = _read_image(template_path, f"Template '{item['label']}'")
        except ValueError:
            detection["status"] = "invalid_template"
            detections.append(detection)
            continue

        match = _match_template(screen, template, config["scales"])
        if match is None:
            detection["status"] = "no_match_region"
            detections.append(detection)
            continue

        detection["score"] = match["score"]
        detection["scale"] = match["scale"]
        detection["box"] = {
            "x": match["x"],
            "y": match["y"],
            "width": match["width"],
            "height": match["height"],
        }
        detection["status"] = (
            "detected" if match["score"] >= item["threshold"]
            else "low_confidence"
        )
        detections.append(detection)

    ambiguous = []
    accepted = [item for item in detections if item["status"] == "detected"]
    for index, first in enumerate(accepted):
        for second in accepted[index + 1:]:
            overlap = _intersection_over_union(first["box"], second["box"])
            if overlap >= 0.3:
                first["status"] = "ambiguous"
                second["status"] = "ambiguous"
                ambiguous.append({
                    "labels": [first["label"], second["label"]],
                    "overlap": round(overlap, 3),
                })

    return {
        "image": str(image_path),
        "dimensions": {
            "width": int(screen.shape[1]),
            "height": int(screen.shape[0]),
        },
        "detections": detections,
        "missing": [
            item["label"] for item in detections
            if item["status"] in (
                "low_confidence", "missing_template", "invalid_template",
                "no_match_region",
            )
        ],
        "ambiguous": ambiguous,
    }


def _print_report(report):
    print("Screenshot: {image} ({width}x{height})".format(
        image=report["image"],
        width=report["dimensions"]["width"],
        height=report["dimensions"]["height"],
    ))
    for item in report["detections"]:
        if item["score"] is None:
            details = item["status"]
        else:
            details = (
                "{status} score={score:.3f} threshold={threshold:.2f} "
                "scale={scale:.2f} box={box}"
            ).format(**item)
        print("  {label}: {details}".format(
            label=item["label"], details=details
        ))
    if report["ambiguous"]:
        for item in report["ambiguous"]:
            print("  AMBIGUOUS: {labels} overlap={overlap}".format(**item))
    else:
        print("  Ambiguous detections: none")
    if report["missing"]:
        print("  Missing detections: {labels}".format(
            labels=", ".join(report["missing"])
        ))
    else:
        print("  Missing detections: none")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Passively recognize configured army items in a static image."
    )
    parser.add_argument("image", help="path to a screenshot image")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="recognition JSON config (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        report = recognize_image(args.image, args.config)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Recognition error: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
