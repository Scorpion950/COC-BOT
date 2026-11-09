import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from loot_analysis import (
    ConfigurationError,
    analyze_screenshot,
    load_config,
    normalize_number,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ABOVE = PROJECT_ROOT / "loot_test_images" / "loot_above_threshold.png"
IMAGE_BELOW = PROJECT_ROOT / "loot_test_images" / "loot_below_threshold.png"
DEFAULT_CONFIG = PROJECT_ROOT / "loot_analysis_config.json"
HAS_RAPIDOCR = (
    importlib.util.find_spec("rapidocr_onnxruntime") is not None
)


class MockOCREngine:
    def __init__(self, gold=None, elixir=None, dark=None, header=True,
                 confidence=0.98, error=None):
        self.gold = gold
        self.elixir = elixir
        self.dark = dark
        self.header = header
        self.confidence = confidence
        self.error = error

    @staticmethod
    def _line(text, center_y, confidence):
        half_height = 12
        return [
            [
                [30, center_y - half_height],
                [240, center_y - half_height],
                [240, center_y + half_height],
                [30, center_y + half_height],
            ],
            text,
            confidence,
        ]

    def __call__(self, image):
        if self.error is not None:
            raise self.error
        height = image.shape[0]
        result = []
        if self.header:
            result.append(self._line("Available Loot:", height * 0.28,
                                     self.confidence))
        if self.gold is not None:
            result.append(self._line(str(self.gold), height * 0.45,
                                     self.confidence))
        if self.elixir is not None:
            result.append(self._line(str(self.elixir), height * 0.675,
                                     self.confidence))
        if self.dark is not None:
            result.append(self._line(str(self.dark), height * 0.90,
                                     self.confidence))
        return result, 0.01


class LootAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp_dir = Path(self.temporary.name)
        self.valid_image = self.temp_dir / "valid.png"
        self.assertTrue(cv2.imwrite(
            str(self.valid_image), np.zeros((1078, 1917, 3), dtype=np.uint8)
        ))

    def analyze(self, gold, elixir, dark=None, confidence=0.98):
        return analyze_screenshot(
            self.valid_image,
            ocr_engine=MockOCREngine(
                gold=gold, elixir=elixir, dark=dark,
                confidence=confidence,
            ),
        )

    def test_both_values_above_threshold_qualify(self):
        result = self.analyze("600,000", "700,000")
        self.assertEqual(result["final_result"], "QUALIFIES")

    def test_gold_below_threshold_does_not_qualify(self):
        result = self.analyze("300,000", "700,000")
        self.assertEqual(result["gold"]["check"], "FAIL")
        self.assertEqual(result["elixir"]["check"], "PASS")
        self.assertEqual(result["final_result"], "DOES NOT QUALIFY")

    def test_elixir_below_threshold_does_not_qualify(self):
        result = self.analyze("700,000", "300,000")
        self.assertEqual(result["gold"]["check"], "PASS")
        self.assertEqual(result["elixir"]["check"], "FAIL")
        self.assertEqual(result["final_result"], "DOES NOT QUALIFY")

    def test_both_values_below_threshold_do_not_qualify(self):
        result = self.analyze("400,000", "400,000")
        self.assertEqual(result["final_result"], "DOES NOT QUALIFY")

    def test_gold_exactly_at_threshold_passes(self):
        result = self.analyze("500,000", "700,000")
        self.assertEqual(result["gold"]["check"], "PASS")
        self.assertEqual(result["final_result"], "QUALIFIES")

    def test_elixir_exactly_at_threshold_passes(self):
        result = self.analyze("700,000", "500,000")
        self.assertEqual(result["elixir"]["check"], "PASS")
        self.assertEqual(result["final_result"], "QUALIFIES")

    def test_both_values_exactly_at_threshold_qualify(self):
        result = self.analyze("500,000", "500,000")
        self.assertEqual(result["final_result"], "QUALIFIES")

    def test_dark_elixir_never_affects_qualification(self):
        result = self.analyze("600,000", "700,000", dark="99,999,999")
        self.assertEqual(result["dark_elixir"], {
            "status": "ignored", "value": None,
        })
        self.assertEqual(result["final_result"], "QUALIFIES")

    def test_grouped_numbers_and_common_ocr_glyph_errors_normalize(self):
        result = self.analyze("5OO,OOO", "5O0 000")
        self.assertEqual(result["gold"]["value"], 500000)
        self.assertEqual(result["elixir"]["value"], 500000)
        self.assertEqual(result["final_result"], "QUALIFIES")
        self.assertEqual(normalize_number("2.657.691"), 2657691)

    def test_ocr_exception_is_inconclusive(self):
        result = analyze_screenshot(
            self.valid_image,
            ocr_engine=MockOCREngine(error=RuntimeError("test OCR failure")),
        )
        self.assertEqual(result["status"], "ocr_error")
        self.assertEqual(result["final_result"], "INCONCLUSIVE")
        self.assertEqual(result["gold"]["status"], "ocr_error")

    def test_missing_gold_is_inconclusive(self):
        result = analyze_screenshot(
            self.valid_image,
            ocr_engine=MockOCREngine(gold=None, elixir="700000"),
        )
        self.assertEqual(result["gold"]["status"], "missing_value")
        self.assertEqual(result["final_result"], "INCONCLUSIVE")

    def test_missing_elixir_is_inconclusive(self):
        result = analyze_screenshot(
            self.valid_image,
            ocr_engine=MockOCREngine(gold="700000", elixir=None),
        )
        self.assertEqual(result["elixir"]["status"], "missing_value")
        self.assertEqual(result["final_result"], "INCONCLUSIVE")

    def test_low_confidence_value_is_inconclusive(self):
        result = self.analyze("700000", "700000", confidence=0.40)
        self.assertEqual(result["gold"]["status"], "low_confidence")
        self.assertEqual(result["final_result"], "INCONCLUSIVE")

    def test_conflicting_ocr_candidates_are_ambiguous(self):
        base_engine = MockOCREngine(gold="700000", elixir="700000")

        def duplicate_gold(image):
            items, elapsed = base_engine(image)
            items.append(MockOCREngine._line(
                "300000", image.shape[0] * 0.45, 0.95
            ))
            return items, elapsed

        result = analyze_screenshot(self.valid_image, ocr_engine=duplicate_gold)
        self.assertEqual(result["gold"]["status"], "ambiguous")
        self.assertEqual(result["final_result"], "INCONCLUSIVE")

    def test_missing_screenshot_is_invalid(self):
        result = analyze_screenshot(
            self.temp_dir / "missing.png",
            ocr_engine=MockOCREngine(gold="700000", elixir="700000"),
        )
        self.assertEqual(result["status"], "invalid_screenshot")
        self.assertEqual(result["final_result"], "INCONCLUSIVE")

    def test_unreadable_screenshot_is_not_zero(self):
        invalid = self.temp_dir / "not_an_image.png"
        invalid.write_text("not an image", encoding="utf-8")
        result = analyze_screenshot(
            invalid,
            ocr_engine=MockOCREngine(gold="700000", elixir="700000"),
        )
        self.assertEqual(result["status"], "unreadable_screenshot")
        self.assertIsNone(result["gold"]["value"])
        self.assertEqual(result["final_result"], "INCONCLUSIVE")

    def test_normalized_region_supports_different_image_resolutions(self):
        image_800 = self.temp_dir / "800x450.png"
        self.assertTrue(cv2.imwrite(
            str(image_800), np.zeros((450, 800, 3), dtype=np.uint8)
        ))
        result = analyze_screenshot(
            image_800,
            ocr_engine=MockOCREngine(gold="600000", elixir="600000"),
        )
        self.assertEqual(result["final_result"], "QUALIFIES")

    def test_invalid_configuration_is_rejected(self):
        config_path = self.temp_dir / "invalid_config.json"
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["comparison_mode"] = "OR"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            load_config(config_path)

    @unittest.skipUnless(HAS_RAPIDOCR, "RapidOCR is unavailable in this Python")
    def test_real_above_threshold_screenshot(self):
        result = analyze_screenshot(IMAGE_ABOVE)
        self.assertEqual(result["gold"]["value"], 2657691)
        self.assertEqual(result["elixir"]["value"], 2210618)
        self.assertEqual(result["final_result"], "QUALIFIES")

    @unittest.skipUnless(HAS_RAPIDOCR, "RapidOCR is unavailable in this Python")
    def test_real_below_threshold_screenshot(self):
        result = analyze_screenshot(IMAGE_BELOW)
        self.assertEqual(result["gold"]["value"], 202431)
        self.assertEqual(result["elixir"]["value"], 209485)
        self.assertEqual(result["final_result"], "DOES NOT QUALIFY")


if __name__ == "__main__":
    unittest.main()
