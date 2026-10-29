"""Tiny CPU native FM -> selection -> PG-only updates -> serving round trip."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import safetensors.torch
import torch

from drvla.sae import SAEConfig, TopKSAE
from experiments.exp2.common import OUTPUT, write_json
from experiments.exp2.test_native_forward import tiny_model
from experiments.exp2.test_pipeline import SyntheticData, SyntheticPolicy
from experiments.exp2.arm1 import pilot, selection, serve_policy
from experiments.exp2.arm1.model import TaskBoundary, frozen_digest, pg_only


class Arm1PipelineTests(unittest.TestCase):
    def test_native_end_to_end(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        with tempfile.TemporaryDirectory(prefix="arm1_technical_", dir=OUTPUT) as directory:
            root = Path(directory)
            run, checkpoint, reference = (root / p for p in ("run", "checkpoint", "reference"))
            for folder in (run, checkpoint, reference):
                folder.mkdir()
            model = tiny_model(num_layers=18, action_dim=32, action_horizon=10)
            safetensors.torch.save_model(model, checkpoint / "model.safetensors")
            model.requires_grad_(False)
            policy, data = SyntheticPolicy(model), SyntheticData()
            saes = root / "saes"
            d = np.full((10, 128), -1., np.float32)
            d[:, :16] = np.arange(1, 17)[None]
            for layer in (5, 11):
                folder = saes / f"paligemma.layer_{layer}.output"
                folder.mkdir(parents=True)
                sae = TopKSAE(SAEConfig(input_dim=16, num_features=128, k=127))
                with torch.no_grad():
                    # Ensure active candidates with a bias-free native encoder.
                    sae.pre_bias[0] = -1000.
                    sae.encoder.weight[:, 0] = 1.
                    sae.encoder.weight[:16, 0] = 2.
                sae.save(folder / "sae.pt")
            for layer in (5, 11, 17):
                np.savez(reference / f"probe_pg{layer}.npz", mean=np.zeros(16), scale=np.ones(16),
                         coef=np.zeros((10, 16)), intercept=np.zeros(10), classes=np.arange(10))
            write_json(run / "manifest.json", {"technical_test": True})
            boundary = TaskBoundary(model)
            with patch.object(selection, "SAES", saes), patch.object(pilot, "SAES", saes), \
                    patch.object(pilot, "CHECKPOINT", checkpoint):
                bases = selection.build(policy, boundary, data, run, device="cpu", statistics={5: d, 11: d})
                info = json.loads((run / "selection.json").read_text())
                for layer in (5, 11):
                    for row in info["layers"][str(layer)]["tasks"]:
                        self.assertEqual(row["overlap_with_rq1"], 8)
                    self.assertEqual(bases["acttask"][layer].shape, bases["random_acttask"][layer].shape)
                pg_only(model)
                frozen = frozen_digest(model)
                results = pilot.train(policy, boundary, data, run, bases, steps=1, validation_limit=10,
                                      device="cpu", reference=reference)
                self.assertEqual(frozen_digest(model), frozen)
                self.assertEqual(len(results), 8)
                self.assertTrue((run / "checkpoint_acttask/model.safetensors").is_file())
            boundary.close()
            safetensors.torch.load_model(model, checkpoint / "model.safetensors")
            model.requires_grad_(False)
            row = data.rows("score")[0]
            raw = data.raw(row)
            observation = {"observation/image": raw["image"], "observation/wrist_image": raw["wrist"],
                           "observation/state": raw["state"], "prompt": row["prompt"]}
            with patch.object(serve_policy, "load_policy", return_value=policy), \
                    patch.object(serve_policy, "CHECKPOINT", checkpoint), \
                    patch.object(serve_policy, "REFERENCE", reference):
                server = serve_policy.RectifiedPolicy(run, device="cpu")

                def infer(arm):
                    return server.infer({**observation, "exp2": {"arm": arm, "noise_seed": 7}})

                original = infer("frozen")["actions"]
                for arm in ("sft_pg_only_off", "acttask_off", "acttask_on",
                            "random_acttask_off", "random_acttask_on"):
                    result = infer(arm)
                    self.assertTrue(np.isfinite(result["actions"]).all())
                    self.assertEqual(result["exp2"]["projection_removed"], arm.endswith("_off"))
                    np.testing.assert_array_equal(original, infer("frozen")["actions"])
                server.boundary.close()


if __name__ == "__main__":
    unittest.main()
