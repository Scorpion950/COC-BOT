import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2

from army_recognition import DEFAULT_CONFIG, recognize_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "reference_images"
TROOP_CARDS = REFERENCE_DIR / "troop_cards.png"
FULL_ARMY_BAR = REFERENCE_DIR / "full_army_bar.png"
WRONG_IMAGE = PROJECT_ROOT / "attack_btn_menu_1.png"


class ArmyRecognitionTests(unittest.TestCase):
    def recognize(self, image, config=DEFAULT_CONFIG):
        return recognize_image(image, config)

    @staticmethod
    def statuses(report):
        return {item["label"]: item["status"] for item in report["detections"]}

    def test_recognizes_troop_cards(self):
        report = self.recognize(TROOP_CARDS)
        statuses = self.statuses(report)
        for label in ("Momma P.E.K.K.A", "Baby Dragons", "Troop Launcher"):
            self.assertEqual(statuses[label], "detected")
        for item in report["detections"]:
            if item["label"] in (
                "Momma P.E.K.K.A", "Baby Dragons", "Troop Launcher"
            ):
                self.assertGreaterEqual(item["score"], item["threshold"])
        self.assertEqual(
            [item["label"] for item in report["detections"]],
            [
                "Momma P.E.K.K.A", "Baby Dragons", "Barbarian King",
                "Archer Queen", "Grand Warden", "Royal Champion",
                "Troop Launcher",
            ],
        )

    def test_recognizes_all_four_heroes(self):
        report = self.recognize(FULL_ARMY_BAR)
        statuses = self.statuses(report)
        for label in (
            "Barbarian King", "Archer Queen", "Grand Warden", "Royal Champion"
        ):
            self.assertEqual(statuses[label], "detected")

    def test_reports_missing_template(self):
        config = self._temporary_config(
            [{"label": "Missing sample", "path": "not-there.png",
              "threshold": 0.8}]
        )
        report = self.recognize(TROOP_CARDS, config)
        self.assertEqual(report["detections"][0]["status"], "missing_template")

    def test_reports_invalid_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_file = root / "not-an-image.png"
            invalid_file.write_text("invalid image", encoding="utf-8")
            config = self._write_config(
                root, [{"label": "Invalid sample", "path": invalid_file.name,
                        "threshold": 0.8}]
            )
            report = self.recognize(TROOP_CARDS, config)
            self.assertEqual(report["detections"][0]["status"], "invalid_template")

    def test_reports_missing_screenshot(self):
        with self.assertRaisesRegex(ValueError, "Screenshot.*not a readable image"):
            self.recognize(PROJECT_ROOT / "missing_screenshot.png")

    def test_reports_incorrect_image_as_low_confidence(self):
        report = self.recognize(WRONG_IMAGE)
        self.assertTrue(all(
            item["status"] != "detected" for item in report["detections"]
        ))

    def test_reports_low_confidence_match(self):
        config = self._temporary_config(
            [{"label": "Momma P.E.K.K.A", "path": "army_templates/momma_pekka.png",
              "threshold": 0.99}]
        )
        report = self.recognize(WRONG_IMAGE, config)
        self.assertEqual(report["detections"][0]["status"], "low_confidence")
        self.assertLess(report["detections"][0]["score"], 0.99)
        self.assertEqual(report["missing"], ["Momma P.E.K.K.A"])

    def test_reports_overlapping_class_matches_as_ambiguous(self):
        config = self._temporary_config([
            {"label": "Class A", "path": "army_templates/momma_pekka.png",
             "threshold": 0.8},
            {"label": "Class B", "path": "army_templates/momma_pekka.png",
             "threshold": 0.8},
        ])
        report = self.recognize(TROOP_CARDS, config)
        self.assertEqual(
            [item["status"] for item in report["detections"]],
            ["ambiguous", "ambiguous"],
        )
        self.assertEqual(report["ambiguous"][0]["labels"], ["Class A", "Class B"])
        self.assertEqual(report["missing"], [])

    def test_recognizes_uniformly_resized_screenshot(self):
        image = cv2.imread(str(FULL_ARMY_BAR), cv2.IMREAD_COLOR)
        resized = cv2.resize(
            image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resized.png"
            self.assertTrue(cv2.imwrite(str(path), resized))
            report = self.recognize(path)
        self.assertTrue(all(
            self.statuses(report)[label] == "detected"
            for label in (
                "Momma P.E.K.K.A", "Baby Dragons", "Troop Launcher",
                "Barbarian King", "Archer Queen", "Grand Warden",
                "Royal Champion",
            )
        ))
        self.assertEqual(report["dimensions"], {"width": 645, "height": 75})

    def test_missing_opencv_is_reported(self):
        with mock.patch("army_recognition._opencv",
                        side_effect=RuntimeError("OpenCV unavailable")):
            with self.assertRaisesRegex(RuntimeError, "OpenCV unavailable"):
                self.recognize(TROOP_CARDS)

    @staticmethod
    def _write_config(directory, templates):
        config = {
            "scales": [0.35, 0.5, 0.75, 1.0, 1.5, 2.0],
            "templates": templates,
        }
        path = Path(directory) / "test_config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _temporary_config(self, templates):
        self.addCleanup(self._cleanup_temporary_config)
        temporary = tempfile.TemporaryDirectory()
        self._temporary_directory = temporary
        root = Path(temporary.name)
        template_dir = root / "army_templates"
        template_dir.mkdir()
        source = PROJECT_ROOT / "army_templates" / "momma_pekka.png"
        if source.is_file():
            (template_dir / source.name).write_bytes(source.read_bytes())
        return self._write_config(root, templates)

    def _cleanup_temporary_config(self):
        temporary = getattr(self, "_temporary_directory", None)
        if temporary is not None:
            temporary.cleanup()
            del self._temporary_directory


if __name__ == "__main__":
    unittest.main()
