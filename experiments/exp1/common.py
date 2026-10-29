"""Isolated Exp1 outputs and read-only provenance helpers (Python 3.8 compatible)."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "workspace/exp1"
SITES = {"pg11": "paligemma.layer_11.output", "ae11": "action_expert.layer_11.output",
         "pg17": "paligemma.layer_17.output"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_path(path):
    path = Path(path).resolve()
    if EXP_ROOT.resolve() not in path.parents:
        raise ValueError(f"Exp1 writes must be below {EXP_ROOT}: {path}")
    return path


def write_json(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, arrays):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def protected_inventory():
    result = {}
    for directory in ("activations", "saes", "index", "checkpoints", "workspace/rq1", "workspace/rq2",
                      "experiments/rq1", "experiments/rq2"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                st = path.stat()
                result[str(path.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
    return result
