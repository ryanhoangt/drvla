"""Step 1: label every logged rollout with what the robot did, without any SAE.

For each episode (``rollout_logger.py`` output):

* ``first_lift``: first object raised > LIFT_DZ above its starting height.
* ``first_release_near``: object/fixture whose position is closest (xy) to the lifted object
  when the gripper first opens after the lift.
* ``trap`` (perturbed suites only):
  - ``_swap``: for each goal object X that was moved, its swap partner Y now sits at X's old
    region. Grasp trap: ``first_lift == Y``. Place trap: the lifted object is released closer to
    Y than to X when X is the receptacle.
  - ``_task``: ``orig_goal_reached`` (the original task was completed), or the first lifted object
    is an original-goal object that is not in the new goal.
* ``trap_t``: env step of the event that decided the label (first lift or release); used as the
  alignment point ("time of commitment") for RQ1.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from task_meta import suite_meta  # noqa: E402

LIFT_DZ = 0.04


def _objects_in(goal):
    return {x for pred in goal for x in pred[1:]}


def _base(name):
    # BDDL goal arguments can be regions of an object (e.g. basket_1_contain_region); map to the body name.
    return name


def label_episode(npz_path, meta, suite):
    d = np.load(npz_path, allow_pickle=False)
    names = [str(n) for n in d["obj_names"]]
    pos = d["obj_pos"]  # (T, n, 3)
    action = d["action"]
    eef = d["eef_pos"]
    T = len(pos)
    z0 = pos[0, :, 2]
    lifted = (pos[:, :, 2] - z0) > LIFT_DZ  # (T, n)
    # Only count objects near the gripper as lifted (drawers/doors move too).
    near = np.linalg.norm(pos - eef[:, None, :], axis=-1) < 0.12
    lifted &= near
    first_lift, lift_t = None, None
    if lifted.any():
        t_idx, o_idx = np.nonzero(lifted)
        k = np.argmin(t_idx)
        first_lift, lift_t = names[o_idx[k]], int(t_idx[k])

    release_near, release_t = None, None
    if first_lift is not None:
        opening = np.nonzero(action[lift_t:, -1] < 0)[0]
        if len(opening):
            release_t = lift_t + int(opening[0])
            i = names.index(first_lift)
            dist = np.linalg.norm(pos[release_t, :, :2] - pos[release_t, i, :2], axis=-1)
            dist[i] = np.inf
            release_near = names[int(np.argmin(dist))]

    out = {"first_lift": first_lift, "lift_t": lift_t, "release_near": release_near, "release_t": release_t,
           "trap": None, "trap_kind": None, "trap_t": None, "T": T}
    pert = suite.rpartition("_")[2]
    goal_objs = {g for g in _objects_in(meta["goal"]) if g in names}

    if pert == "swap":
        moved = meta["moved"]
        # partner of X: the moved object whose new region is X's old region
        partner = {}
        for x, (old_x, _new_x) in moved.items():
            for y, (_old_y, new_y) in moved.items():
                if y != x and new_y == old_x:
                    partner[x] = y
        grasp_targets = [x for x in partner if x in goal_objs and x in names]
        trap, kind, tt = False, None, lift_t
        for x in grasp_targets:
            y = partner[x]
            if first_lift == y:
                trap, kind = True, "grasp_decoy"
            elif first_lift is not None and first_lift != x and release_near == y and x in names:
                # x is the receptacle: placed on the object now at x's old spot
                trap, kind, tt = True, "place_decoy", release_t
        out.update(trap=trap, trap_kind=kind, trap_t=tt if trap else (lift_t if lift_t is not None else None))
        out["swap_relevant"] = bool(grasp_targets)
    elif pert == "task":
        orig_objs = {g for g in _objects_in(meta["orig_goal"]) if g in names}
        new_objs = goal_objs
        wrong = orig_objs - new_objs
        orig_done = bool(np.asarray(d["orig_goal_ok"]).all(-1).any()) if d["orig_goal_ok"].size else False
        trap = orig_done or (first_lift in wrong)
        kind = "orig_goal" if orig_done else ("grasp_orig_obj" if first_lift in wrong else None)
        out.update(trap=bool(trap), trap_kind=kind, trap_t=lift_t, orig_goal_reached=orig_done)
    return out


def label_suite(rollout_dir, suite):
    rollout_dir = Path(rollout_dir) / suite
    metas = suite_meta(suite)
    results = [json.loads(l) for l in (rollout_dir / "results.jsonl").read_text().splitlines()]
    rows = []
    for r in results:
        npz = rollout_dir / f"task{r['task_id']:02d}_ep{r['episode']:03d}.npz"
        if not npz.exists():
            continue
        meta = metas.get(r["task"] + ".bddl", {"goal": [], "orig_goal": [], "moved": {}})
        rows.append({**r, **label_episode(npz, meta, suite)})
    return rows


if __name__ == "__main__":
    root, suites = sys.argv[1], sys.argv[2:]
    for suite in suites:
        if not (Path(root) / suite / "results.jsonl").exists():
            continue
        rows = label_suite(root, suite)
        out = Path(root) / suite / "labels.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        n = len(rows)
        succ = sum(r["success"] for r in rows)
        fails = [r for r in rows if not r["success"]]
        traps = sum(bool(r.get("trap")) for r in fails)
        print(f"{suite}: n={n} success={succ} failures={len(fails)} trap_in_failures={traps}")
