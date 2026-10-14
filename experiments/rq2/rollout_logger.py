"""Paired LIBERO-PRO task pilot. Run with the LIBERO Python 3.8 environment.

The policy server runs in openpi's Python 3.11 environment. Both observations,
explicit diffusion noise, simulator initial state, feature edits, and physical
outcomes are saved in workspace/rq2. No RQ1 artifacts are modified.
"""

import argparse
import collections
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

# robosuite otherwise attempts to open another user's /tmp/robosuite.log.
_FileHandler = logging.FileHandler


class _OwnFileHandler(_FileHandler):
    def __init__(self, filename, *args, **kwargs):
        if str(filename) == "/tmp/robosuite.log":
            filename = str(Path(tempfile.gettempdir()) / f"rq2_robosuite_{os.getuid()}.log")
        super().__init__(filename, *args, **kwargs)


logging.FileHandler = _OwnFileHandler

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.envs.bddl_utils import robosuite_parse_problem  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

from experiments.rq2.common import manifest, output_path, sha256, write_json  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]
DEFAULT_CONDITIONS = [
    "baseline", "suppress_old", "boost_new", "patch_old",
    "random_suppress", "general_suppress", "random_boost", "general_boost", "old_prompt",
]


def seed_for(*values):
    payload = json.dumps(values, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    denominator = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def policy_observation(obs, prompt):
    images = {}
    for key, source in [("observation/image", "agentview_image"),
                        ("observation/wrist_image", "robot0_eye_in_hand_image")]:
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        images[key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    return {
        **images, "prompt": prompt,
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32),
    }


def observation_hash(observation):
    digest = hashlib.sha256()
    for key in ("observation/image", "observation/wrist_image", "observation/state"):
        digest.update(observation[key].tobytes())
    return digest.hexdigest()


def goal_values(inner, goal):
    # Fail loudly for invalid predicates rather than quietly changing the labels.
    return np.array([bool(inner._eval_predicate([p[0].lower(), *p[1:]])) for p in goal], dtype=bool)


def goal_reached(values):
    return bool(len(values) and values.all())


def reset_episode(env, initial_state, seed):
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY)
    return obs


def request(client, observation, pair_key, mode, noise_seed, alpha=1.0, active=True):
    return client.infer({**observation, "rq2": {
        "pair": pair_key, "mode": mode, "noise_seed": noise_seed, "alpha": alpha, "active": active,
    }})


def save_npz(path, arrays):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def verify(client, observation, pair, pair_key, noise_seed, alpha, destination):
    """Same-observation causal check and exact identity/passive-hook gates."""
    responses = {}
    for mode in ("unhooked", "baseline", "identity"):
        responses[mode] = request(client, observation, pair_key, mode, noise_seed, alpha)
    repeated = request(client, observation, pair_key, "baseline", noise_seed, alpha)
    reference = responses["baseline"]["actions"]
    checks = {}
    for name, actions in [
        ("unhooked", responses["unhooked"]["actions"]),
        ("identity", responses["identity"]["actions"]), ("repeat", repeated["actions"]),
    ]:
        checks[name] = float(np.max(np.abs(actions - reference)))
        np.testing.assert_array_equal(actions, reference, err_msg=f"{name} changed baseline actions")
    old_obs = {**observation, "prompt": pair["old_prompt"]}
    responses["old_prompt"] = request(client, old_obs, pair_key, "baseline", noise_seed, alpha)
    old_direction = (responses["old_prompt"]["actions"][:5, :3] - reference[:5, :3]).ravel()
    old_norm = np.linalg.norm(old_direction)
    effects = {}
    for mode in DEFAULT_CONDITIONS:
        if mode in responses:
            continue
        responses[mode] = request(client, observation, pair_key, mode, noise_seed, alpha)
    for mode, response in responses.items():
        diff = (response["actions"][:5, :3] - reference[:5, :3]).ravel()
        diff_norm = np.linalg.norm(diff)
        effects[mode] = {
            "action_xyz_delta_norm": float(diff_norm),
            "alignment_with_old_prompt": float(np.dot(diff, old_direction) / (diff_norm * old_norm))
            if diff_norm > 1e-8 and old_norm > 1e-8 else None,
            **response["rq2"].get("diagnostics", {}),
        }
        if effects[mode].get("min_edited_feature", 0) < -1e-6:
            raise AssertionError(f"Negative SAE feature coordinates in {mode}")
    arrays = {"image": observation["observation/image"], "wrist": observation["observation/wrist_image"],
              "state": observation["observation/state"], "noise": responses["baseline"]["rq2"]["noise"]}
    for mode, response in responses.items():
        arrays["action/" + mode] = response["actions"]
        for key in ("features", "features_after", "feature_delta"):
            if key in response["rq2"]:
                arrays[mode + "/" + key] = response["rq2"][key]
    save_npz(destination.with_suffix(".npz"), arrays)
    report = {"noise_seed": noise_seed, "observation_sha256": observation_hash(observation),
              "identity_checks_max_abs": checks, "effects": effects}
    write_json(destination.with_suffix(".json"), report)
    logging.info("PASS identity gates; %s", json.dumps(checks))
    return report


def rollout(client, env, obs, initial_state, pair, pair_key, condition, noise_base,
            args, expected_observation_hash, destination):
    inner = env.env
    prompt = pair["old_prompt"] if condition == "old_prompt" else pair["new_prompt"]
    mode = "baseline" if condition == "old_prompt" else condition
    old_goal = pair["old_goal"]
    new_goal = inner.parsed_problem["goal_state"]
    if observation_hash(policy_observation(obs, prompt)) != expected_observation_hash:
        raise AssertionError("Paired initial observations differ between conditions")
    names = list(inner.objects_dict) + list(inner.fixtures_dict)
    start_positions = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names])
    old_started = goal_reached(goal_values(inner, old_goal))
    old_ever = old_started
    new_ever = goal_reached(goal_values(inner, new_goal))
    plan = collections.deque()
    log = collections.defaultdict(list)
    first_lift, lift_t, first_old_t, first_new_t = None, None, None, None
    count = 0
    frames = []
    started = time.monotonic()
    for t in range(args.max_steps):
        if not plan:
            observation = policy_observation(obs, prompt)
            noise_seed = seed_for(noise_base, pair_key, args.current_episode, count)
            response = request(client, observation, pair_key, mode, noise_seed, args.alpha,
                               count < args.intervene_calls or args.intervene_calls < 0)
            info = response["rq2"]
            log["infer_t"].append(t)
            log["chunk"].append(response["actions"])
            log["image"].append(observation["observation/image"])
            log["wrist_image"].append(observation["observation/wrist_image"])
            log["state"].append(observation["observation/state"])
            log["noise"].append(info["noise"])
            log["noise_seed"].append(noise_seed)
            for key in ("pooled", "features", "features_after", "edited_features", "feature_delta"):
                log[key].append(info[key])
            for key, value in info["diagnostics"].items():
                if key != "feature_ids":
                    log["edit/" + key].append(value)
            # IDs may differ during adaptive control matching; pad to dictionary width.
            ids = np.full(args.k, -1, dtype=np.int32)
            chosen = info["diagnostics"]["feature_ids"]
            ids[:len(chosen)] = chosen
            log["edit/feature_ids"].append(ids)
            if count == 0:
                save_npz(destination.with_name(destination.stem + "_initial.npz"), {
                    "sim_state": np.asarray(inner.sim.get_state().flatten()), "initial_state": initial_state,
                    "image": observation["observation/image"], "wrist_image": observation["observation/wrist_image"],
                })
            plan.extend(response["actions"][: args.replan_steps])
            count += 1
        action = plan.popleft()
        log["action"].append(action)
        log["eef_pos"].append(obs["robot0_eef_pos"].copy())
        log["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
        if args.video:
            frames.append(policy_observation(obs, prompt)["observation/image"])
        obs, _, done, _ = env.step(np.asarray(action).tolist())
        log["eef_pos_after"].append(obs["robot0_eef_pos"].copy())
        log["gripper_qpos_after"].append(obs["robot0_gripper_qpos"].copy())
        positions = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names])
        log["obj_pos"].append(positions)
        eef = obs["robot0_eef_pos"]
        lifted = ((positions[:, 2] - start_positions[:, 2]) > 0.04) & (
            np.linalg.norm(positions - eef[None, :], axis=1) < 0.12)
        if first_lift is None and lifted.any():
            first_lift, lift_t = names[int(np.flatnonzero(lifted)[0])], t + 1
        old_values, new_values = goal_values(inner, old_goal), goal_values(inner, new_goal)
        log["old_goal_ok"].append(old_values)
        log["new_goal_ok"].append(new_values)
        if goal_reached(old_values):
            old_ever = True
            if first_old_t is None:
                first_old_t = t + 1
        if goal_reached(new_values):
            new_ever = True
            if first_new_t is None:
                first_new_t = t + 1
        desired_done = goal_reached(old_values) if condition == "old_prompt" else bool(done)
        if desired_done:
            break
    old_objects = {arg for pred in old_goal for arg in pred[1:] if arg in names}
    new_objects = {arg for pred in new_goal for arg in pred[1:] if arg in names}
    lifted_old = first_lift in old_objects - new_objects
    lifted_new = first_lift in new_objects - old_objects
    success = old_ever if condition == "old_prompt" else new_ever
    # Keep events and terminal failure labels separate; allow recovery after old-task events.
    old_event = bool((old_ever and not old_started) or lifted_old)
    trap_failure = bool(not success and old_event and condition != "old_prompt")
    arrays = {key: np.asarray(value) for key, value in log.items()}
    arrays.update(obj_names=np.asarray(names), initial_object_positions=start_positions,
                  initial_state=initial_state)
    save_npz(destination, arrays)
    if args.video:
        import imageio
        imageio.mimwrite(str(destination.with_suffix(".mp4")), frames, fps=10)
    record = {
        "suite": args.suite, "task_id": args.current_task, "task": pair_key.split("/", 1)[1],
        "episode": args.current_episode, "noise_base": noise_base, "condition": condition,
        "prompt": prompt, "success": success, "new_goal_reached": new_ever, "old_goal_reached": old_ever,
        "old_goal_started_true": old_started, "old_task_event": old_event, "trap_failure": trap_failure,
        "first_lift": first_lift, "lift_t": lift_t, "lifted_old_target": bool(lifted_old),
        "lifted_new_target": bool(lifted_new), "first_old_goal_t": first_old_t, "first_new_goal_t": first_new_t,
        "steps": t + 1, "n_infer": count, "n_edits": int(sum(log["edit/applied"])),
        "mean_delta_norm": float(np.mean(log["edit/delta_norm"])),
        "seconds": time.monotonic() - started, "observation_sha256": expected_observation_hash,
        "artifact": str(destination.relative_to(args.run_dir)),
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object_task")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--intervene-calls", type=int, default=6)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.suite != "libero_object_task":
        raise ValueError("This first pilot is restricted to libero_object_task")
    if args.max_steps < 1 or not 1 <= args.replan_steps <= 10:
        raise ValueError("Invalid rollout horizon or replan interval")
    if any(c not in DEFAULT_CONDITIONS and c != "identity" for c in args.conditions):
        raise ValueError("Unknown condition")
    args.run_dir = output_path(args.run_dir)
    bundle = json.loads(args.features.read_text())
    args.k = bundle["manifest"]["k"]
    config = {
        "format": "rq2-rollouts-v1", "suite": args.suite, "tasks": args.tasks, "episodes": args.episodes,
        "noise_seeds": args.noise_seeds, "conditions": args.conditions, "alpha": args.alpha,
        "intervene_calls": args.intervene_calls, "replan_steps": args.replan_steps,
        "max_steps": args.max_steps, "video": args.video,
        "features_sha256": sha256(args.features), "source_sha256": sha256(Path(__file__)),
    }
    manifest(args.run_dir / "clients" / ("tasks_" + "_".join(map(str, args.tasks)) + ".json"), config)
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    metadata = client.get_server_metadata()
    if metadata.get("sae_sha256") != bundle["manifest"]["sae_sha256"]:
        raise ValueError("Server and client have different SAE provenance")
    if metadata.get("features_sha256") != sha256(args.features):
        raise ValueError("Server and client have different feature/prompt bundles")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    root = Path(get_libero_path("bddl_files"))
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        pair_key = "libero_object/" + task.name
        pair = bundle["pairs"][pair_key]
        bddl = root / task.problem_folder / task.bddl_file
        if sha256(bddl) != bundle["manifest"]["bddl_sha256"][str(bddl.relative_to(root))]:
            raise ValueError("Simulator BDDL differs from frozen feature provenance")
        parsed = robosuite_parse_problem(str(bddl))
        actual_goal = parsed["goal_state"]
        canonical_goal = [[p[0].lower(), *p[1:]] for p in pair["new_goal"]]
        if list(map(list, actual_goal)) != canonical_goal:
            raise AssertionError("Feature pair goal and simulator goal differ")
        if bundle["manifest"].get("prompt_style") == "env-lowercase":
            if pair["new_prompt"] != " ".join(parsed["language_instruction"]):
                raise AssertionError("Canonical prompt differs from simulator language")
            original_bddl = root / "libero_object" / task.bddl_file
            original = robosuite_parse_problem(str(original_bddl))
            if pair["old_prompt"] != " ".join(original["language_instruction"]):
                raise AssertionError("Canonical donor prompt differs from original BDDL language")
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        states = suite.get_task_init_states(task_id)
        task_dir = args.run_dir / "rollouts" / f"task{task_id:02d}"
        results = task_dir / "results.jsonl"
        task_dir.mkdir(parents=True, exist_ok=True)
        completed = {}
        if results.exists():
            for line in results.read_text().splitlines():
                r = json.loads(line)
                key = (r["episode"], r["noise_base"], r["condition"])
                if key in completed:
                    raise ValueError("Duplicate result on resume")
                if not (args.run_dir / r["artifact"]).exists():
                    raise FileNotFoundError(r["artifact"])
                completed[key] = r
        try:
            for episode in args.episodes:
                args.current_task, args.current_episode = task_id, episode
                sim_seed = seed_for("simulator", task_id, episode)
                for noise_base in args.noise_seeds:
                    obs = reset_episode(env, states[episode], sim_seed)
                    observation = policy_observation(obs, pair["new_prompt"])
                    destination = args.run_dir / "verification" / f"task{task_id:02d}_ep{episode:03d}_seed{noise_base}"
                    initial_noise_seed = seed_for(noise_base, pair_key, episode, 0)
                    report = verify(client, observation, pair, pair_key, initial_noise_seed, args.alpha, destination)
                    if args.verify_only:
                        continue
                    # A deterministic shuffle balances condition order, without changing paired noise.
                    order = list(args.conditions)
                    np.random.default_rng(seed_for("order", task_id, episode, noise_base)).shuffle(order)
                    for condition in order:
                        key = (episode, noise_base, condition)
                        if key in completed:
                            continue
                        obs = reset_episode(env, states[episode], sim_seed)
                        artifact = task_dir / f"ep{episode:03d}_seed{noise_base}_{condition}.npz"
                        record = rollout(client, env, obs, states[episode], pair, pair_key, condition,
                                         noise_base, args, report["observation_sha256"], artifact)
                        with results.open("a") as stream:
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            stream.flush()
                        completed[key] = record
                        logging.info("task %d ep %d seed %d %s success=%s old_event=%s first_lift=%s %.1fs",
                                     task_id, episode, noise_base, condition, record["success"],
                                     record["old_task_event"], record["first_lift"], record["seconds"])
        finally:
            env.close()


if __name__ == "__main__":
    main()
