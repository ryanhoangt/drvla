"""Technical CPU integration of score -> files -> probes -> SFT -> report.

All inputs/weights are synthetic, tiny and temporary. This is a software test,
not an empirical CIFT result. The production OpenPI forward and Exp2 pipeline
functions execute; no production checkpoint or rollout is modified.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import safetensors.torch
import torch

from drvla.sae import SAEConfig, TopKSAE
from openpi import transforms
from openpi.shared.normalize import NormStats

from experiments.exp2 import pilot
from experiments.exp2.analyze import analyze
from experiments.exp2.artifacts import load_scores
from experiments.exp2.common import OUTPUT, write_json
from experiments.exp2.model import Boundary, tensors
from experiments.exp2.test_native_forward import tiny_model


class SyntheticData:
    def __init__(self):
        self.manifest = {"prompts": [f"task{i}" for i in range(10)]}
        self.manifest["counterfactual_candidates"] = {
            p: [q for q in self.manifest["prompts"] if q != p] for p in self.manifest["prompts"]}

    def rows(self, split):
        repeats = 4 if split == "score" else 2
        return [{"task": t, "prompt": f"task{t}", "row": i, "frame": i,
                 "episode": t*100+i//2+(20 if split == "validation" else 0)}
                for t in range(10) for i in range(repeats)]

    def raw(self, row):
        rng = np.random.default_rng(row["episode"]*10+row["frame"])
        return {"image": rng.integers(0, 256, (2, 2, 3), dtype=np.uint8),
                "wrist": rng.integers(0, 256, (2, 2, 3), dtype=np.uint8),
                "state": np.zeros(32, np.float32), "actions": rng.normal(size=(10, 32)).astype(np.float32)}


class SyntheticPolicy:
    def __init__(self, model):
        self._model = model
        self.normalizer = transforms.Normalize({"actions": NormStats(mean=np.zeros(32), std=np.ones(32))})

    def _input_transform(self, raw):
        task = int(raw["prompt"].removeprefix("task"))
        return self.normalizer({"actions": raw["actions"], "state": raw["observation/state"],
            "image": {"base": raw["observation/image"], "wrist": raw["observation/wrist_image"],
                      "masked": np.zeros_like(raw["observation/image"])},
            "image_mask": {"base": np.True_, "wrist": np.True_, "masked": np.False_},
            "tokenized_prompt": np.array([task+1, 1], np.int32), "tokenized_prompt_mask": np.ones(2, bool)})

    @torch.no_grad()
    def infer(self, observation, noise):
        raw = {"image": observation["observation/image"], "wrist": observation["observation/wrist_image"],
               "state": observation["observation/state"], "actions": np.zeros((10, 32), np.float32)}
        obs, _ = tensors(self, raw, observation["prompt"], device="cpu")
        action = self._model.sample_actions(torch.device("cpu"), obs, noise=torch.tensor(noise)[None])
        return {"actions": action[0, :, :7].numpy()}


class PipelineTests(unittest.TestCase):
    def test_synthetic_native_pipeline_round_trip(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        OUTPUT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="technical_test_", dir=OUTPUT) as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            model = tiny_model(num_layers=18, action_dim=32, action_horizon=10)
            safetensors.torch.save_model(model, checkpoint / "model.safetensors")
            model.requires_grad_(False)
            policy, data = SyntheticPolicy(model), SyntheticData()
            sae_root = root / "saes"
            for layer in (5, 11, 17):
                directory = sae_root / f"paligemma.layer_{layer}.output"
                directory.mkdir(parents=True)
                TopKSAE(SAEConfig(input_dim=16, num_features=128, k=48)).save(directory / "sae.pt")
            write_json(run / "manifest.json", {"job_id": "CPU_SOFTWARE_TEST", "partition": "CPU",
                       "gpu": "NONE: synthetic temporary test", "arguments": {"steps": 1}})
            boundary = Boundary(model, (5, 11, 17))
            with patch.object(pilot, "SAES", sae_root), patch.object(pilot, "CHECKPOINT", checkpoint):
                bases = pilot.score(policy, boundary, data, run, device="cpu")
                for layer in (5, 11, 17):
                    saved = load_scores(run / f"score_pg{layer}.npz")
                    self.assertEqual(saved["sample_sensitivity"].shape, (40, 128))
                    self.assertEqual(saved["sensitivity"].shape, (128,))
                    np.testing.assert_allclose(saved["sample_sensitivity"].mean(0), saved["sensitivity"])
                pilot.fit_probes(run)
                results = pilot.train(policy, boundary, data, run, bases, steps=1, validation_limit=10,
                                      arms=["sft", "sae_m", "random_sae", "pca_promptdiff"], device="cpu")
                self.assertEqual(set(results), {"frozen", "sft", "sae_m", "random_sae", "pca_promptdiff"})
            boundary.close()
            analyze(run)
            self.assertTrue((run / "EXP2_report.md").is_file())
            self.assertTrue((run / "figures/f1_scoring.png").is_file())
            self.assertTrue((run / "figures/f2_validation.png").is_file())
            summary = json.loads((run / "figure_data.json").read_text())
            self.assertFalse(summary["evaluation_complete"])
            self.assertEqual(summary["completed_rollouts"], 0)


if __name__ == "__main__":
    unittest.main()
