"""Copy only committed rollout records into a fresh evaluation directory.

Pure standard library; shared with the separate Python 3.8 simulator.
"""

import itertools
import json
from pathlib import Path
import shutil

from experiments.exp2.common import destination, sha256, write_json


PROTOCOL_KEYS = ("protocol", "training_run", "task_ids", "suites", "episodes", "noise_base", "arms",
                 "training_prompts", "horizon", "replan_steps", "explicit_hook_modes")


def result_key(row):
    return row["suite"], row["task_id"], row["episode"], row["arm"]


def validate_results(config, rows, directory):
    expected = set(itertools.product(config["suites"], config["task_ids"], config["episodes"], config["arms"]))
    seen = set()
    hashes = {}
    directory = Path(directory).resolve()
    for row in rows:
        key = result_key(row)
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate completed rollout: {key}")
        if row["noise_base"] != config["noise_base"]:
            raise ValueError("Completed rollout noise differs from protocol")
        artifact = Path(row["artifact"])
        if artifact.is_absolute() or ".." in artifact.parts:
            raise ValueError("Unsafe completed artifact path")
        expected_name = f"rollouts/{row['suite']}/task{row['task_id']:02d}_ep{row['episode']:03d}_{row['arm']}.npz"
        if str(artifact) != expected_name or not (directory / artifact).is_file():
            raise ValueError(f"Missing or mismatched completed artifact: {artifact}")
        case_key = f"{row['suite']}/{row['task_id']}/{row['episode']}"
        digest = row["initial_observation_sha256"]
        if case_key in hashes and hashes[case_key] != digest:
            raise ValueError("Committed arms disagree on initial observation hash")
        hashes[case_key] = digest
        seen.add(key)
    return hashes


def seed_results(config, previous):
    """Keep the source untouched; byte-verify copied completed NPZs."""
    previous = Path(previous).resolve()
    target = destination(config["eval_dir"])
    if target == previous or previous in target.parents or target in previous.parents:
        raise ValueError("Resume output must be separate from source evaluation")
    old_config = json.loads((previous / "config.json").read_text())
    for key in PROTOCOL_KEYS:
        if old_config.get(key) != config.get(key):
            raise ValueError(f"Resume protocol mismatch: {key}")
    rows = json.loads((previous / "results.json").read_text())
    validate_results(config, rows, previous)
    provenance = []
    for row in rows:
        source, output = previous / row["artifact"], target / row["artifact"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        shutil.copy2(source, output)
        digest = sha256(source)
        if digest != sha256(output):
            raise ValueError(f"Copied rollout differs: {source}")
        provenance.append({"key": list(result_key(row)), "artifact": row["artifact"], "sha256": digest})
    write_json(target / "results.json", rows)
    write_json(target / "resume_provenance.json", {"source_evaluation": str(previous),
               "source_config_sha256": sha256(previous / "config.json"),
               "source_results_sha256": sha256(previous / "results.json"),
               "reused_rollouts": len(rows), "copied_artifacts": provenance})
    return len(rows)


def read_progress(config):
    directory = Path(config["eval_dir"])
    path = directory / "results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    return rows, validate_results(config, rows, directory)
