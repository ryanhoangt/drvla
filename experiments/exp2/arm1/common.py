"""Protect all previous scientific artifacts, including Exp2 job 60924."""

from experiments.exp2.common import ROOT, inventory

REFERENCE = ROOT / "workspace/exp2/cift_job_60924"


def protected():
    result = inventory()
    for folder in (REFERENCE, ROOT / "workspace/exp2/data_v1", ROOT / "experiments/exp2"):
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                stat = path.stat()
                result[str(path.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    return result
