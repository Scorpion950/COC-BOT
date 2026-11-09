import tempfile
import unittest
from pathlib import Path

import cv2

from next_button_recognition import recognize_next_button


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABOVE_SCREENSHOT = (
    PROJECT_ROOT / "loot_test_images" / "loot_above_threshold.png"
)
BELOW_SCREENSHOT = (
    PROJECT_ROOT / "loot_test_images" / "loot_below_threshold.png"
)
NO_BUTTON_IMAGE = PROJECT_ROOT / "reference_images" / "troop_cards.png"


class NextButtonRecognitionTests(unittest.TestCase):
    def test_detects_next_button_in_first_opponent_screenshot(self):
        result = recognize_next_button(ABOVE_SCREENSHOT)
        self.assertTrue(result["detected"])
        self.assertEqual(result["status"], "detected")
        self.assertGreaterEqual(result["confidence"], result["threshold"])
        self.assertIsNotNone(result["box"])

    def test_detects_next_button_in_second_opponent_screenshot(self):
        result = recognize_next_button(BELOW_SCREENSHOT)
        self.assertTrue(result["detected"])
        self.assertGreaterEqual(result["confidence"], result["threshold"])

    def test_reports_negative_on_image_without_next_button(self):
        image = cv2.imread(str(ABOVE_SCREENSHOT), cv2.IMREAD_COLOR)
        image[772:892, 1656:1912] = image[720:840, 1180:1436]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same-size-no-next.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            result = recognize_next_button(path)
        self.assertFalse(result["detected"])
        self.assertEqual(result["status"], "not_detected")
        self.assertIsNotNone(result["confidence"])
        self.assertLess(result["confidence"], result["threshold"])

    def test_reports_negative_on_unrelated_reference(self):
        result = recognize_next_button(NO_BUTTON_IMAGE)
        self.assertFalse(result["detected"])

    def test_detects_button_after_resolution_change(self):
        image = cv2.imread(str(ABOVE_SCREENSHOT), cv2.IMREAD_COLOR)
        resized = cv2.resize(
            image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "half-size.png"
            self.assertTrue(cv2.imwrite(str(path), resized))
            result = recognize_next_button(path)
        self.assertTrue(result["detected"])
        self.assertEqual(result["scale"], 0.5)


if __name__ == "__main__":
    unittest.main()
