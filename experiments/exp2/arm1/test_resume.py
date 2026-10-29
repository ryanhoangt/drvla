"""Resume safety checks without importing any model, CUDA or simulator."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp2.common import OUTPUT, sha256, write_json
from experiments.exp2.arm1.resume import read_progress, seed_results, validate_results


class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        previous, target = root / "old", root / "new"
        previous.mkdir()
        target.mkdir()
        config = {"protocol": "exp2-arm1-v1", "training_run": "/same/frozen/training",
                  "task_ids": [0], "suites": ["libero_object"], "episodes": [0], "noise_base": 7,
                  "arms": ["acttask_off", "acttask_on"], "training_prompts": ["test prompt"],
                  "horizon": 280, "replan_steps": 5, "explicit_hook_modes": True,
                  "eval_dir": str(previous)}
        artifact = "rollouts/libero_object/task00_ep000_acttask_off.npz"
        source = previous / artifact
        source.parent.mkdir(parents=True)
        source.write_bytes(b"completed-array-content")
        row = {"suite": "libero_object", "task_id": 0, "episode": 0, "arm": "acttask_off",
               "noise_base": 7, "artifact": artifact, "initial_observation_sha256": "same-image-state"}
        write_json(previous / "config.json", config)
        write_json(previous / "results.json", [row])
        return previous, target, dict(config, eval_dir=str(target)), row

    def test_copy_only_completed_and_preserve_source(self):
        with tempfile.TemporaryDirectory(prefix="arm1_resume_test_", dir=OUTPUT) as directory:
            previous, target, config, row = self.fixture(Path(directory))
            original = {p.relative_to(previous): sha256(p) for p in previous.rglob("*") if p.is_file()}
            # Uncommitted NPZ must not be reused after an interrupted rollout.
            (previous / "rollouts/libero_object/task00_ep000_acttask_on.npz").write_bytes(b"partial")
            self.assertEqual(seed_results(config, previous), 1)
            rows, hashes = read_progress(config)
            self.assertEqual(rows, [row])
            self.assertEqual(hashes["libero_object/0/0"], "same-image-state")
            self.assertFalse((target / "rollouts/libero_object/task00_ep000_acttask_on.npz").exists())
            for relative, digest in original.items():
                self.assertEqual(sha256(previous / relative), digest)
            self.assertNotEqual((previous / row["artifact"]).stat().st_ino, (target / row["artifact"]).stat().st_ino)
            self.assertEqual(json.loads((target / "resume_provenance.json").read_text())["reused_rollouts"], 1)

    def test_protocol_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix="arm1_resume_test_", dir=OUTPUT) as directory:
            previous, target, config, _ = self.fixture(Path(directory))
            config["noise_base"] = 8
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                seed_results(config, previous)
            self.assertFalse((target / "results.json").exists())

    def test_duplicate_or_missing_artifacts_rejected(self):
        with tempfile.TemporaryDirectory(prefix="arm1_resume_test_", dir=OUTPUT) as directory:
            previous, _, config, row = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_results(config, [row, row], previous)
            (previous / row["artifact"]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                validate_results(config, [row], previous)


if __name__ == "__main__":
    unittest.main()
