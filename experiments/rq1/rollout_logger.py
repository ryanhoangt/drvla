"""LIBERO / LIBERO-PRO rollouts that log activations, actions and simulator state.

Same rollout protocol as ``openpi/examples/libero/main.py`` (replan every 5 steps,
10 wait steps, 180-degree image rotation, prompt from the BDDL file for the
``_lan`` / ``_task`` suites). Talks to ``serve_policy_acts.py``. Per episode it writes
``<out>/<suite>/task{task_id:02d}_ep{episode:03d}.npz`` with

* per inference call: ``infer_t`` (env step), ``act/<layer>`` (d,), ``chunk`` (10, 7)
* per env step: ``eef_pos`` (3,), ``gripper_qpos`` (2,), ``action`` (7,),
  ``obj_pos`` (n_obj, 3) for ``obj_names``, ``goal_ok`` and ``orig_goal_ok`` (per predicate)

and appends one line to ``<out>/<suite>/results.jsonl``.

Run with the LIBERO python (3.8), from the openpi directory:
    LIBERO_CONFIG_PATH=../workspace/libero_pro_config PYTHONPATH=$PWD/third_party/LIBERO-PRO:$PWD/../experiments/rq1 \
    MUJOCO_GL=egl python ../experiments/rq1/rollout_logger.py --suite libero_object_swap --port 8000 --out ../workspace/rq1
"""

import argparse
import collections
import json
import logging
import math
import pathlib

import os
import tempfile

# robosuite hard-codes /tmp/robosuite.log, which may belong to another user on a shared machine.
_FileHandler = logging.FileHandler


class _RedirectedFileHandler(_FileHandler):
    def __init__(self, filename, *args, **kwargs):
        if str(filename) == "/tmp/robosuite.log":
            filename = os.path.join(tempfile.gettempdir(), f"robosuite_{os.getuid()}.log")
        super().__init__(filename, *args, **kwargs)


logging.FileHandler = _RedirectedFileHandler

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
PERTURBATIONS = ("lan", "object", "swap", "task", "env")


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def split_suite(suite):
    base, _, pert = suite.rpartition("_")
    if pert in PERTURBATIONS and base in MAX_STEPS:
        return base, pert
    return suite, None


def eval_goal(inner, goal_state):
    out = []
    for state in goal_state:
        try:
            out.append(bool(inner._eval_predicate(state)))
        except Exception:  # predicate refers to an object missing from this scene
            out.append(False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--tasks", type=int, nargs="*")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    np.random.seed(args.seed)

    base, pert = split_suite(args.suite)
    max_steps = MAX_STEPS[base]
    prompt_from_bddl = pert in ("lan", "task")
    out_dir = pathlib.Path(args.out) / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done_eps = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done_eps.add((r["task_id"], r["episode"]))

    suite = benchmark.get_benchmark_dict()[args.suite]()
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    task_ids = args.tasks if args.tasks else range(suite.n_tasks)

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = bddl_root / task.problem_folder / task.bddl_file
        orig_bddl = bddl_root / base / task.bddl_file
        orig_goal = robosuite_parse_problem(str(orig_bddl))["goal_state"] if orig_bddl.exists() else []
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        prompt = env.language_instruction if prompt_from_bddl else task.language
        inner = env.env
        goal = inner.parsed_problem["goal_state"]
        obj_names = list(inner.objects_dict) + list(inner.fixtures_dict)

        for ep in range(args.trials):
            if (task_id, ep) in done_eps:
                continue
            env.reset()
            obs = env.set_init_state(init_states[ep])
            plan = collections.deque()
            log = collections.defaultdict(list)
            frames = []
            t, success = 0, False
            while t < max_steps + 10:
                if t < 10:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                if args.video:
                    frames.append(img)
                if not plan:
                    state = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )
                    resp = client.infer(
                        {"observation/image": img, "observation/wrist_image": wrist,
                         "observation/state": state, "prompt": str(prompt)}
                    )
                    log["infer_t"].append(t - 10)
                    log["chunk"].append(np.asarray(resp["actions"], dtype=np.float32))
                    for layer, value in resp["activations"].items():
                        log["act/" + layer].append(np.asarray(value, dtype=np.float32))
                    plan.extend(resp["actions"][:5])
                action = plan.popleft()
                log["eef_pos"].append(obs["robot0_eef_pos"].copy())
                log["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
                log["action"].append(np.asarray(action, dtype=np.float32))
                log["obj_pos"].append(np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in obj_names]))
                log["goal_ok"].append(eval_goal(inner, goal))
                log["orig_goal_ok"].append(eval_goal(inner, orig_goal) if orig_goal else [])
                obs, _, done, _ = env.step(np.asarray(action).tolist())
                t += 1
                if done:
                    success = True
                    break

            # State after the last action (the step that may have completed the task).
            final_goal = eval_goal(inner, goal)
            final_orig = eval_goal(inner, orig_goal) if orig_goal else []
            log["orig_goal_ok"].append(final_orig)
            arrays = {k: np.asarray(v) for k, v in log.items()}
            arrays["final_goal_ok"] = np.asarray(final_goal)
            arrays["obj_names"] = np.asarray(obj_names)
            np.savez_compressed(out_dir / f"task{task_id:02d}_ep{ep:03d}.npz", **arrays)
            if args.video:
                imageio.mimwrite(out_dir / f"task{task_id:02d}_ep{ep:03d}_{success}.mp4", frames, fps=10)
            orig_ok = bool(np.asarray(log["orig_goal_ok"]).all(-1).any()) if orig_goal else None
            rec = {"suite": args.suite, "task_id": task_id, "task": task.name, "prompt": prompt, "episode": ep,
                   "success": success, "steps": t - 10, "orig_goal_reached": orig_ok}
            with open(results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            logging.info("%s task %d ep %d success=%s orig_goal=%s steps=%d", args.suite, task_id, ep, success, orig_ok, t - 10)
        env.close()


if __name__ == "__main__":
    main()
