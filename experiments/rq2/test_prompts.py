import unittest

from experiments.rq2.normalize_feature_prompts import canonical_bundle


class PromptTests(unittest.TestCase):
    def test_canonicalizes_only_prompt_text_and_provenance(self):
        source = {"manifest": {"format": "v1", "sae_sha256": "frozen"},
                  "pairs": {"p": {"old_prompt": "Pick The Milk", "new_prompt": "Pick  the Tomato Sauce",
                                  "old": [1, 2], "new": [3, 4], "new_mean": [0.1, 0.2]}}}
        result = canonical_bundle(source, "source-hash")
        self.assertEqual(result["pairs"]["p"]["new_prompt"], "pick the tomato sauce")
        self.assertEqual(source["pairs"]["p"]["new_prompt"], "Pick  the Tomato Sauce")
        for key in ("old", "new", "new_mean"):
            self.assertEqual(result["pairs"]["p"][key], source["pairs"]["p"][key])
        self.assertEqual(result["manifest"]["sae_sha256"], "frozen")
        self.assertEqual(result["manifest"]["source_features_sha256"], "source-hash")


if __name__ == "__main__":
    unittest.main()
