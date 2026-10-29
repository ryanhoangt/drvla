"""Paired ID/task/swap evaluation. Run only in the separate LIBERO Python 3.8 env."""

import argparse
import collections
import json
import logging
from pathlib import Path
import time

import numpy as np

from experiments.exp2.common import destination, sha256, write_json
from experiments.rq1.task_meta import BDDL_ROOT, parse, suite_meta
from experiments.rq2.rollout_logger import (OffScreenRenderEnv, WebsocketClientPolicy, benchmark,
                                          goal_reached, goal_values, observation_hash,
                                          policy_observation, reset_episode, robosuite_parse_problem, seed_for)


def save_npz(path, **arrays):
    path = destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def request(client, observation, arm, noise):
    return client.infer({**observation, "exp2": {"arm": arm, "noise_seed": noise}})


def rollout(client, env, initial, case, arm, prompts, run, expected_hash, noise_base):
    task_id, episode, suite = case["task_id"], case["episode"], case["suite"]
    seed = seed_for("simulator", task_id, episode)
    obs = reset_episode(env, initial, seed)
    observation = policy_observation(obs, case["prompt"])
    if observation_hash(observation) != expected_hash:
        raise RuntimeError("Paired simulator initial observations differ")
    inner = env.env
    names = list(inner.objects_dict) + list(inner.fixtures_dict)
    initial_positions = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names])
    log, plan = collections.defaultdict(list), collections.deque()
    first_lift, lift_t, old_ever, success = None, None, False, False
    started = time.monotonic()
    inference = 0
    key = "libero_object/" + case["task_name"]
    for step in range(280):
        if not plan:
            observation = policy_observation(obs, case["prompt"])
            noise_seed = seed_for(noise_base, key, episode, inference)
            result = request(client, observation, arm, noise_seed)
            plan.extend(result["actions"][:5])
            log["infer_t"].append(step)
            log["image"].append(observation["observation/image"])
            log["wrist_image"].append(observation["observation/wrist_image"])
            log["state"].append(observation["observation/state"])
            log["chunk"].append(result["actions"])
            log["noise_seed"].append(noise_seed)
            for layer in (5, 11, 17):
                for kind in ("pooled", "probe"):
                    log[f"{kind}_pg{layer}"].append(result["exp2"][f"{kind}_pg{layer}"])
            inference += 1
        action = np.asarray(plan.popleft())
        obs, _, _, _ = env.step(action.tolist())
        positions = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names])
        eef = obs["robot0_eef_pos"].copy()
        log["action"].append(action)
        log["obj_pos"].append(positions)
        log["eef_pos_after"].append(eef)
        log["gripper_qpos_after"].append(obs["robot0_gripper_qpos"].copy())
        old = goal_values(inner, case["old_goal"])
        new = goal_values(inner, case["new_goal"])
        log["orig_goal_ok"].append(old)
        log["goal_ok"].append(new)
        old_ever = old_ever or goal_reached(old)
        success = success or goal_reached(new)
        lifted = ((positions[:, 2]-initial_positions[:, 2]) > 0.04) & (
            np.linalg.norm(positions-eef[None], axis=1) < 0.12)
        candidates = [i for i in np.flatnonzero(lifted) if names[int(i)] in inner.objects_dict]
        if first_lift is None and candidates:
            first_lift, lift_t = names[int(candidates[0])], step+1
        if success:
            break
    output = run / "rollouts" / suite / f"task{task_id:02d}_ep{episode:03d}_{arm}.npz"
    arrays = {key: np.asarray(value) for key, value in log.items()}
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite evaluation array: {name}")
    save_npz(output, **arrays, initial_object_positions=initial_positions,
             initial_state=np.asarray(initial), obj_names=np.asarray(names))
    old_object, new_object = case["old_object"], case["new_object"]
    event = False
    if suite.endswith("_task"):
        event = (first_lift == old_object and old_object != new_object) or old_ever
    elif suite.endswith("_swap"):
        event = first_lift in case["old_location_decoys"]
    original_prompt = case["task_name"].replace("_", " ")
    new_prompt = "pick up the " + new_object.rsplit("_", 1)[0].replace("_", " ") + " and place it in the basket"
    old_index, new_index = prompts.index(original_prompt), prompts.index(new_prompt)
    return {"suite": suite, "task_id": task_id, "task_name": case["task_name"], "episode": episode,
            "arm": arm, "noise_base": noise_base, "prompt": case["prompt"], "success": bool(success),
            "trap_event": bool(event), "trap_failure": bool(event and not success),
            "first_lift": first_lift, "lift_t": lift_t, "correct_first_lift": first_lift == new_object,
            "old_object": old_object, "new_object": new_object, "old_location_decoys": case["old_location_decoys"],
            "steps": step+1, "seconds": time.monotonic()-started,
            "initial_observation_sha256": expected_hash, "artifact": str(output.relative_to(run)),
            "t0_probe": {str(layer): {"p_original": float(log[f"probe_pg{layer}"][0][old_index]),
                                       "p_instructed": float(log[f"probe_pg{layer}"][0][new_index])}
                         for layer in (5, 11, 17)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = destination(config["eval_dir"])
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    metadata = client.get_server_metadata()
    if metadata.get("manifest_sha256") != sha256(Path(config["training_run"]) / "manifest.json"):
        raise ValueError("Wrong model server")
    if not metadata.get("projection_removed"):
        raise ValueError("Primary inference must remove training projections")
    results, hashes = [], {}
    prompts = config["training_prompts"]
    for suite_name in config["suites"]:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        metas = suite_meta(suite_name)
        for task_id in config["task_ids"]:
            task = suite.get_task(task_id)
            path = BDDL_ROOT / suite_name / (task.name + ".bddl")
            parsed = robosuite_parse_problem(str(path))
            prompt = " ".join(parsed["language_instruction"])
            old = parse(BDDL_ROOT / "libero_object" / (task.name + ".bddl"))
            new = parse(path)
            old_object, new_object = old["goal"][0][1], new["goal"][0][1]
            moved = metas[task.name+".bddl"]["moved"]
            old_region = moved.get(new_object, [None, None])[0]
            decoys = [name for name, regions in moved.items()
                      if name != new_object and old_region is not None and regions[1] == old_region]
            for episode in config["episodes"]:
                case = {"task_id": task_id, "task_name": task.name, "suite": suite_name,
                        "episode": episode, "prompt": prompt, "old_goal": old["goal"],
                        "new_goal": new["goal"], "old_object": old_object, "new_object": new_object,
                        "old_location_decoys": decoys}
                if parsed["goal_state"] != [[g[0].lower(), *g[1:]] for g in case["new_goal"]]:
                    raise ValueError("Simulator goal mismatch")
                env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256, camera_widths=256)
                initial = suite.get_task_init_states(task_id)[episode]
                try:
                    obs = reset_episode(env, initial, seed_for("simulator", task_id, episode))
                    observation = policy_observation(obs, prompt)
                    expected = observation_hash(observation)
                    hashes[f"{suite_name}/{task_id}/{episode}"] = expected
                    # Exact checkpoint round-trip checks at the same state/noise.
                    seed = seed_for(config["noise_base"], "libero_object/"+task.name, episode, 0)
                    a = request(client, observation, "frozen", seed)["actions"]
                    request(client, observation, "sft", seed)
                    b = request(client, observation, "frozen", seed)["actions"]
                    np.testing.assert_array_equal(a, b, err_msg="Checkpoint switching contaminated frozen baseline")
                    for arm in config["arms"]:
                        result = rollout(client, env, initial, case, arm, prompts, run, expected, config["noise_base"])
                        results.append(result)
                        write_json(run / "results.json", results)
                        logging.info("RESULT %s", json.dumps(result))
                finally:
                    env.close()
    write_json(run / "gates.json", {"paired_initial_hashes": hashes, "checkpoint_round_trip_exact": True,
                                    "projection_removed": True})
    write_json(run / "status.json", {"status": "complete", "rollouts": len(results)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
