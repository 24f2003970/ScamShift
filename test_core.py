"""Offline tests for metric arithmetic, fixture boundaries and drift."""
from collections import Counter
import unittest

from scamshift.datasets import audit_splits, binary_review, examples, overlaps_evaluation
from scamshift.drift import compare, distance, input_features, priority_band
from scamshift.evaluation import metrics, quality, sliced_metrics
from scamshift.text_features import normalize, robust_normalize, script


class EvaluationTest(unittest.TestCase):
    def test_threshold_confusion_and_tied_ranking_scores(self):
        result = metrics([0, 1, 0, 1], [.1, .5, .5, .9], .5)
        self.assertEqual(result["confusion"], {"tp": 2, "fp": 1, "fn": 0, "tn": 1})
        self.assertEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["f1"], .8)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["roc_auc"], .875)
        self.assertLess(result["recall_ci95"][0], result["recall"])

    def test_missing_classes_return_undefined_rates(self):
        result = metrics([0, 0], [.1, .2], .5)
        for key in ("recall", "recall_ci95", "precision", "roc_auc", "average_precision"):
            self.assertIsNone(result[key])
        self.assertEqual(result["false_alarm"], 0)
        self.assertIsNone(metrics([], [], .5)["accuracy"])

    def test_script_slices_preserve_support(self):
        rows = [{"script": "Latin", "label": 1}, {"script": "Other", "label": 0}]
        groups = sliced_metrics(rows, [.4, .9], .5, "script")
        self.assertEqual(groups["Latin"]["missed"], 1)
        self.assertEqual(groups["Other"]["false_flags"], 1)


class DatasetTest(unittest.TestCase):
    def test_cohorts_and_template_boundaries(self):
        self.assertEqual([len(examples(split)) for split in ("review", "validation", "audit", "holdout", "robustness")],
                         [30, 12, 24, 24, 10])
        for pair in audit_splits():
            self.assertEqual(pair["template_matches"], [], pair)
            if {pair["left"], pair["right"]} != {"review", "audit"}:
                self.assertEqual(pair["shared_families"], [], pair)
        self.assertTrue(overlaps_evaluation(examples("holdout")[0]["text"].upper() + "!!!"))
        self.assertFalse(overlaps_evaluation("Please collect the library scanner from reception."))

    def test_feedback_labels_and_opt_in_normalization(self):
        self.assertEqual(binary_review(2), 1)
        self.assertIsNone(binary_review(None))
        self.assertEqual(normalize("  ＯＴＰ   entry  "), "otp entry")
        self.assertEqual(robust_normalize("O.T.P at www.test.example").strip(), "otp at  urltoken")
        self.assertEqual(script("OTP की सूचना"), "Mixed Latin and Devanagari")


class DriftTest(unittest.TestCase):
    def test_score_bands_follow_the_configured_threshold(self):
        self.assertEqual(priority_band(.60, .60), "Flagged")
        self.assertEqual(priority_band(.56, .60), "Near threshold")
        self.assertEqual(priority_band(.45, .60), "Below threshold")

    def test_jensen_shannon_distance_and_no_sample_case(self):
        self.assertEqual(distance(Counter(a=2), Counter(a=50)), 0)
        self.assertEqual(distance(Counter(a=2), Counter(b=3)), 1)
        self.assertIsNone(distance(Counter(), Counter(a=2)))

    def test_input_features_do_not_contain_raw_message(self):
        text = "private example www.demo.example"
        features = input_features(text)
        self.assertEqual(features["url"], "Yes")
        self.assertNotIn("private", repr(features))
        self.assertNotIn("www.demo.example", repr(features))

    def test_prediction_drift_compares_same_model_version(self):
        def row(model, text, score):
            return {"model_id": model, "features": input_features(text), "score": score, "flagged": "No"}
        before = [row(1, "Short message", "< 0.2")] * 10
        after = [row(2, "Payment notification at www.demo.example", "≥ 0.8")] * 10
        report = {item["feature"]: item for item in compare(before, after, active=2, minimum=10)}
        self.assertEqual(report["URL present"]["status"], "significant")
        self.assertEqual(report["Score distribution"]["status"], "insufficient")
        self.assertEqual(report["Model version mix"]["kind"], "operational")

    def test_url_distribution_covers_all_drift_states(self):
        def batch(url_count, total):
            texts = ["Synthetic case at www.demo.example"] * url_count + ["Synthetic case"] * (total - url_count)
            return [{"model_id": 1, "features": input_features(text), "score": "< 0.2", "flagged": "No"}
                    for text in texts]

        reference = batch(10, 20)
        for count, total, expected in ((4, 8, "insufficient"), (10, 20, "normal"),
                                       (16, 20, "warning"), (20, 20, "significant")):
            with self.subTest(expected=expected):
                report = compare(reference, batch(count, total), active=1, minimum=10)
                result = next(feature for feature in report if feature["feature"] == "URL present")
                self.assertEqual(result["status"], expected)


class PromotionTest(unittest.TestCase):
    def test_rewards_lower_false_alarms_at_equal_recall(self):
        benchmark = metrics([0, 1] * 12, [.2, .8] * 12, .5)
        batch = metrics([0, 1] * 12, [.2, .8] * 11 + [.8, .8], .5)
        active = {"benchmark": benchmark, "campaign_audit": batch}
        improved = {"benchmark": benchmark, "campaign_audit": benchmark}
        self.assertTrue(all(gate["pass"] for gate in quality(improved, active)))
        self.assertFalse(all(gate["pass"] for gate in quality(active, active)))
        regressed = metrics([0, 1] * 12, [.2, .8] * 11 + [.2, .2], .5)
        self.assertFalse(all(gate["pass"] for gate in quality({**improved, "benchmark": regressed}, active)))

    def test_missing_or_tiny_cohorts_fail_closed(self):
        tiny = metrics([0, 1], [.2, .8], .5)
        self.assertFalse(quality({"benchmark": tiny}, {})[0]["pass"])


if __name__ == "__main__":
    unittest.main()
