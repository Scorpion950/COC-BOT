"""Passive OCR analysis of available Gold and Elixir in static screenshots."""

import argparse
import json
import math
import re
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "loot_analysis_config.json"
_NUMBER_TRANSLATION = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5", "B": "8",
})
_NUMBER_SEPARATORS = re.compile(r"[\s,.'’\u00a0\u202f]")


class ConfigurationError(ValueError):
    """Raised when the loot-analysis configuration is invalid."""


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
    """Load configuration without silently replacing invalid settings."""
    config_path = Path(config_path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON in configuration file {config_path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a JSON object")
    for key in ("minimum_gold", "minimum_elixir"):
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0):
            raise ConfigurationError(
                f"'{key}' must be a non-negative integer"
            )
    if config.get("ignore_dark_elixir") is not True:
        raise ConfigurationError("'ignore_dark_elixir' must be true")
    if config.get("comparison_mode") != "AND":
        raise ConfigurationError(
            "'comparison_mode' must be 'AND'; Gold and Elixir must both pass"
        )

    ocr = config.get("ocr")
    if not isinstance(ocr, dict):
        raise ConfigurationError("'ocr' must be a JSON object")
    confidence = ocr.get("minimum_confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ConfigurationError(
            "'ocr.minimum_confidence' must be between 0 and 1"
        )
    upscale = ocr.get("upscale")
    if (isinstance(upscale, bool) or not isinstance(upscale, int)
            or not 1 <= upscale <= 5):
        raise ConfigurationError("'ocr.upscale' must be an integer from 1 to 5")
    if ocr.get("preprocessing") not in ("color", "grayscale"):
        raise ConfigurationError(
            "'ocr.preprocessing' must be 'color' or 'grayscale'"
        )

    region = ocr.get("available_loot_region")
    if not isinstance(region, dict) or set(region) != {
        "x", "y", "width", "height"
    }:
        raise ConfigurationError(
            "'ocr.available_loot_region' must define x, y, width, and height"
        )
    normalized_region = {}
    for key, value in region.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ConfigurationError(
                f"'ocr.available_loot_region.{key}' must be numeric"
            )
        normalized_region[key] = float(value)
    if (normalized_region["x"] < 0 or normalized_region["y"] < 0
            or normalized_region["width"] <= 0
            or normalized_region["height"] <= 0
            or normalized_region["x"] + normalized_region["width"] > 1
            or normalized_region["y"] + normalized_region["height"] > 1):
        raise ConfigurationError(
            "'ocr.available_loot_region' must fit within normalized image bounds"
        )

    for row_name in ("gold_row", "elixir_row"):
        row = ocr.get(row_name)
        if (not isinstance(row, list) or len(row) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in row)
                or not 0 <= row[0] < row[1] <= 1):
            raise ConfigurationError(
                f"'ocr.{row_name}' must be [start, end] within 0..1"
            )
    if ocr["gold_row"][1] > ocr["elixir_row"][0]:
        raise ConfigurationError("Gold and Elixir OCR rows must not overlap")
    return config


def normalize_number(text):
    """Convert OCR digits with common glyph confusions and grouping marks."""
    if not isinstance(text, str):
        return None
    corrected = text.translate(_NUMBER_TRANSLATION).strip()
    if not corrected or re.search(r"[^0-9\s,.'’\u00a0\u202f]", corrected):
        return None
    digits = _NUMBER_SEPARATORS.sub("", corrected)
    if not digits or not digits.isdigit():
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _empty_resource(status):
    return {
        "value": None,
        "raw_text": None,
        "confidence": None,
        "status": status,
        "check": None,
    }


def _base_result(image_path, config):
    return {
        "screenshot": str(image_path),
        "status": "inconclusive",
        "section_status": "not_checked",
        "gold": _empty_resource("not_checked"),
        "elixir": _empty_resource("not_checked"),
        "dark_elixir": {"status": "ignored", "value": None},
        "final_result": "INCONCLUSIVE",
        "thresholds": {
            "minimum_gold": config["minimum_gold"],
            "minimum_elixir": config["minimum_elixir"],
            "comparison_mode": "AND",
        },
        "errors": [],
    }


def _make_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR is required for loot analysis. Install project "
            "dependencies with: pip install -r requirements.txt"
        ) from exc
    return RapidOCR()


def _region_pixels(region, width, height):
    left = max(0, min(width - 1, int(round(region["x"] * width))))
    top = max(0, min(height - 1, int(round(region["y"] * height))))
    right = max(left + 1, min(
        width, int(round((region["x"] + region["width"]) * width))
    ))
    bottom = max(top + 1, min(
        height, int(round((region["y"] + region["height"]) * height))
    ))
    return left, top, right, bottom


def _prepare_region(image, config):
    cv2 = _opencv()
    ocr_config = config["ocr"]
    height, width = image.shape[:2]
    left, top, right, bottom = _region_pixels(
        ocr_config["available_loot_region"], width, height
    )
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Configured Available Loot region is empty")
    if ocr_config["preprocessing"] == "grayscale":
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if ocr_config["upscale"] > 1:
        crop = cv2.resize(
            crop,
            None,
            fx=ocr_config["upscale"],
            fy=ocr_config["upscale"],
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def _line_center_y(item):
    try:
        points = item[0]
        ys = [float(point[1]) for point in points]
        if not ys or not all(math.isfinite(y) for y in ys):
            return None
        return sum(ys) / len(ys)
    except (IndexError, TypeError, ValueError):
        return None


def _analyze_resource(items, row, image_height, minimum_confidence, threshold):
    candidates = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        center_y = _line_center_y(item)
        if center_y is None:
            continue
        row_position = center_y / image_height
        if not row[0] <= row_position <= row[1]:
            continue
        text = str(item[1]).strip()
        value = normalize_number(text)
        if value is None:
            continue
        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            continue
        candidates.append({
            "value": value,
            "raw_text": text,
            "confidence": confidence,
        })

    if not candidates:
        result = _empty_resource("missing_value")
    elif len({item["value"] for item in candidates}) > 1:
        result = _empty_resource("ambiguous")
        result["raw_text"] = " | ".join(item["raw_text"] for item in candidates)
        result["confidence"] = min(item["confidence"] for item in candidates)
    else:
        candidate = max(candidates, key=lambda item: item["confidence"])
        result = dict(candidate)
        result["status"] = (
            "recognized" if candidate["confidence"] >= minimum_confidence
            else "low_confidence"
        )
        result["check"] = None
        if result["status"] == "recognized":
            result["check"] = "PASS" if result["value"] >= threshold else "FAIL"
    return result


def _has_available_loot_header(items, minimum_confidence):
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = re.sub(r"[^a-z]", "", str(item[1]).lower())
        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            continue
        if ("availableloot" in text
                and math.isfinite(confidence)
                and confidence >= minimum_confidence):
            return True
    return False


def analyze_screenshot(image_path, config_path=DEFAULT_CONFIG, ocr_engine=None):
    """Read available Gold and Elixir from a static screenshot."""
    config = load_config(config_path)
    image_path = Path(image_path)
    result = _base_result(image_path, config)
    if not image_path.is_file():
        result["status"] = "invalid_screenshot"
        result["errors"].append(f"Screenshot file not found: {image_path}")
        return result

    cv2 = _opencv()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        result["status"] = "unreadable_screenshot"
        result["errors"].append(f"Could not decode screenshot: {image_path}")
        return result

    try:
        engine = ocr_engine if ocr_engine is not None else _make_ocr_engine()
    except RuntimeError as exc:
        result["status"] = "ocr_unavailable"
        result["errors"].append(str(exc))
        result["gold"] = _empty_resource("ocr_unavailable")
        result["elixir"] = _empty_resource("ocr_unavailable")
        return result

    try:
        prepared = _prepare_region(image, config)
        ocr_result, _ = engine(prepared)
        items = ocr_result or []
    except Exception as exc:
        result["status"] = "ocr_error"
        result["errors"].append(f"OCR failed: {exc}")
        result["gold"] = _empty_resource("ocr_error")
        result["elixir"] = _empty_resource("ocr_error")
        return result

    minimum_confidence = float(config["ocr"]["minimum_confidence"])
    result["section_status"] = (
        "found" if _has_available_loot_header(items, minimum_confidence)
        else "not_found"
    )
    if result["section_status"] == "not_found":
        result["errors"].append(
            "Available Loot heading was not confidently recognized"
        )

    prepared_height = prepared.shape[0]
    result["gold"] = _analyze_resource(
        items, config["ocr"]["gold_row"], prepared_height,
        minimum_confidence, config["minimum_gold"],
    )
    result["elixir"] = _analyze_resource(
        items, config["ocr"]["elixir_row"], prepared_height,
        minimum_confidence, config["minimum_elixir"],
    )

    if result["section_status"] == "found":
        if result["gold"]["status"] != "recognized":
            result["gold"]["check"] = None
        if result["elixir"]["status"] != "recognized":
            result["elixir"]["check"] = None
        if (result["gold"]["status"] == "recognized"
                and result["elixir"]["status"] == "recognized"):
            result["final_result"] = (
                "QUALIFIES"
                if (result["gold"]["value"] >= config["minimum_gold"]
                    and result["elixir"]["value"] >= config["minimum_elixir"])
                else "DOES NOT QUALIFY"
            )
            result["status"] = "complete"
        else:
            result["final_result"] = "INCONCLUSIVE"
            result["status"] = "inconclusive"
    else:
        result["gold"]["check"] = None
        result["elixir"]["check"] = None
        result["status"] = "inconclusive"
    return result


def _format_value(resource):
    if resource["value"] is None:
        return "RECOGNITION FAILED"
    return f"{resource['value']:,}"


def print_report(result):
    """Print a concise human-readable report for the CLI."""
    print(f"GOLD: {_format_value(result['gold'])}")
    print(f"ELIXIR: {_format_value(result['elixir'])}")
    print("DARK ELIXIR: IGNORED")
    for name in ("gold", "elixir"):
        resource = result[name]
        if resource["check"] is not None:
            print(f"{name.upper()} CHECK: {resource['check']}")
        else:
            confidence = resource["confidence"]
            suffix = (
                f", confidence={confidence:.3f}"
                if confidence is not None else ""
            )
            print(
                f"{name.upper()} CHECK: INCONCLUSIVE "
                f"({resource['status']}{suffix})"
            )
    print(f"FINAL RESULT: {result['final_result']}")
    for warning in result["errors"]:
        print(f"WARNING: {warning}")
    for name in ("gold", "elixir"):
        resource = result[name]
        if resource["confidence"] is not None:
            print(
                f"{name.upper()} OCR: {resource['raw_text']!r} "
                f"(confidence={resource['confidence']:.3f}, "
                f"status={resource['status']})"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Passively analyze available Gold and Elixir in a screenshot."
    )
    parser.add_argument("image", help="path to a static screenshot")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="analysis JSON config (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        result = analyze_screenshot(args.image, args.config)
    except (ConfigurationError, OSError, ValueError, RuntimeError) as exc:
        print(f"Configuration or analysis error: {exc}", file=sys.stderr)
        return 2
    print_report(result)
    return 0 if result["final_result"] != "INCONCLUSIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
