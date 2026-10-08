"""Per-task description of what each LIBERO-PRO perturbation changes, from the BDDL files.

For every task of a perturbed suite (``<base>_swap`` / ``<base>_task``) this compares the
BDDL file with the base-suite file of the same name and returns the instruction, the goal
predicates of both files, and the objects whose initial region changed.
"""

import json
import re
import sys
from pathlib import Path

BDDL_ROOT = Path(__file__).resolve().parents[2] / "openpi/third_party/LIBERO-PRO/libero/libero/bddl_files"
BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _section(text, name):
    start = text.find(f"(:{name}")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "("
        depth -= text[i] == ")"
        if depth == 0:
            return text[start : i + 1]
    return text[start:]


def parse(path):
    text = Path(path).read_text()
    language = re.search(r"\(:language ([^)]*)\)", text).group(1).strip()
    init = {}
    for pred, obj, region in re.findall(r"\((On|In) (\S+) (\S+)\)", _section(text, "init")):
        init[obj] = region
    goal = [list(g) for g in re.findall(r"\((\w+) (\S+?)(?: (\S+?))?\)", _section(text, "goal").replace("(And", ""))]
    goal = [[x for x in g if x] for g in goal]
    return {"language": language, "init": init, "goal": goal}


def suite_meta(suite):
    base = next(b for b in BASE_SUITES if suite.startswith(b + "_") or suite == b)
    out = {}
    for path in sorted((BDDL_ROOT / suite).glob("*.bddl")):
        pert, orig = parse(path), parse(BDDL_ROOT / base / path.name)
        moved = sorted(o for o in pert["init"] if orig["init"].get(o) not in (None, pert["init"][o]))
        out[path.name] = {
            "language": pert["language"],
            "orig_language": orig["language"],
            "goal": pert["goal"],
            "orig_goal": orig["goal"],
            "moved": {o: [orig["init"][o], pert["init"][o]] for o in moved},
        }
    return out


if __name__ == "__main__":
    for suite in sys.argv[1:]:
        print(f"===== {suite}")
        for name, m in suite_meta(suite).items():
            print(json.dumps({"task": name[:-5], **m}))
