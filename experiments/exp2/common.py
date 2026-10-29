"""Exp2 output isolation, provenance and shared paths."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "workspace/exp2"
DATASET = Path("/mnt/data/vhoangth2/hf_cache/lerobot/physical-intelligence/libero")
CHECKPOINT = ROOT / "checkpoints/pi05_libero_pytorch"
SAES = ROOT / "saes/pi05_libero_server"


def destination(path):
    path = Path(path).resolve()
    if OUTPUT.resolve() not in path.parents:
        raise ValueError(f"Exp2 outputs must be below {OUTPUT}: {path}")
    return path


def write_json(path, value):
    path = destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    result = {}
    for folder in ("experiments/rq1", "experiments/rq2", "experiments/exp1", "workspace/rq1",
                   "workspace/rq2", "workspace/exp1", "activations", "saes", "index", "checkpoints"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                stat = path.stat()
                result[str(path.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    return result
