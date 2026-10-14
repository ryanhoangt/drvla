"""Artifact provenance and output isolation for the RQ2 pilot."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RQ2_ROOT = ROOT / "workspace" / "rq2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(path):
    """Only write experiment artifacts beneath workspace/rq2, including on resume."""
    path = Path(path).resolve()
    if path == RQ2_ROOT.resolve() or RQ2_ROOT.resolve() not in path.parents:
        raise ValueError(f"RQ2 outputs must be in a subdirectory of {RQ2_ROOT}: {path}")
    return path


def write_json(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest(path, data):
    """Resume only an identical configuration; never replace a different experiment."""
    path = output_path(path)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Configuration differs from {path}; use a new run directory")
    else:
        write_json(path, data)
