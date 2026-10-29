"""One Exp1 object-choice pilot; use the LIBERO Python 3.8 environment."""

import argparse
import collections
import json
import logging
from pathlib import Path
import time

import numpy as np

# Read-only reuse of RQ2 preprocessing/reset helpers; never call its writers.
# Its robosuite log redirection uses this job's isolated TMPDIR.
from experiments.rq2.rollout_logger import (
    OffScreenRenderEnv, WebsocketClientPolicy, benchmark, goal_reached, goal_values,
    observation_hash, policy_observation, reset_episode, robosuite_parse_problem, seed_for,
)
from experiments.exp1.common import ROOT, output_path, save_npz, sha256, write_json


def request(client, obs, mode, site, noise_seed):
    return client.infer({**obs, "exp1": {"mode": mode, "site": site, "noise_seed": noise_seed}})


def preflight(client, obs, config, destination):
    seed = seed_for(config["noise_base"], config["pair_key"], config["episode"], 0)
    responses = {}
    for mode in ("unhooked", "baseline", "self"):
        responses[mode] = request(client, obs, mode, "pg11", seed)
    responses["repeat"] = request(client, obs, "baseline", "pg11", seed)
    reference = responses["baseline"]["actions"]
    checks = {}
    for mode in ("unhooked", "self", "repeat"):
        checks[mode] = float(np.abs(responses[mode]["actions"] - reference).max())
        np.testing.assert_array_equal(responses[mode]["actions"], reference, err_msg=mode)
    responses["old_prompt"] = request(client, {**obs, "prompt": config["old_prompt"]}, "baseline", "pg11", seed)
    contrast = (responses["old_prompt"]["actions"][:5, :3] - reference[:5, :3]).ravel()
    contrast_sq = float(contrast @ contrast)
    if contrast_sq < 1e-10:
        raise ValueError("No initial old/new action contrast; stop instead of a rollout sweep")
    probes = {}
    for site in ("pg11", "ae11", "pg17"):
        modes = ("dense", "pooled", "pooled_norm") if site != "pg17" else ("dense",)
        for mode in modes:
            response = request(client, obs, mode, site, seed)
            name = site + "_" + mode
            responses[name] = response
            diff = (response["actions"][:5, :3] - reference[:5, :3]).ravel()
            probes[name] = {
                "old_action_projection": float(diff @ contrast / contrast_sq),
                "xyz_action_delta_norm": float(np.linalg.norm(diff)),
                "full_action_max_abs_delta": float(np.abs(response["actions"] - reference).max()),
                "diagnostics": response["exp1"]["diagnostics"],
                "recipient_valid_prefix_tokens": response["exp1"]["recipient_valid_prefix_tokens"],
                "donor_valid_prefix_tokens": response["exp1"]["donor_valid_prefix_tokens"],
            }
    # PG17 output is discarded after its own K/V cache has already been built.
    np.testing.assert_array_equal(responses["pg17_dense"]["actions"], reference,
                                  err_msg="PG17 negative connectivity control unexpectedly changed actions")
    site = max(("pg11", "ae11"), key=lambda s: probes[s + "_dense"]["old_action_projection"])
    selected = site if probes[site + "_dense"]["old_action_projection"] >= config["min_dense_projection"] else None
    report = {"identity_max_abs": checks, "observation_sha256": observation_hash(obs),
              "noise_seed": seed, "old_new_contrast_norm": float(np.sqrt(contrast_sq)), "probes": probes,
              "selected_site": selected, "selection_rule": "largest initial dense old-action projection; minimum 0.05"}
    arrays = {"image": obs["observation/image"], "wrist_image": obs["observation/wrist_image"],
              "state": obs["observation/state"], "noise": responses["baseline"]["exp1"]["noise"]}
    for name, response in responses.items():
        arrays["action/" + name] = response["actions"]
    save_npz(destination.with_suffix(".npz"), arrays)
    write_json(destination.with_suffix(".json"), report)
    logging.info("Identity gates PASS; PG17 negative control PASS; selected site=%s", selected)
    return report


def rollout(client, env, obs, initial_state, config, mode, site, expected_hash, destination):
    inner = env.env
    prompt = config["old_prompt"] if mode == "baseline_old" else config["new_prompt"]
    patch_mode = "baseline" if mode.startswith("baseline_") else mode
    if observation_hash(policy_observation(obs, prompt)) != expected_hash:
        raise ValueError("Paired initial inputs differ")
    names = list(inner.objects_dict) + list(inner.fixtures_dict)
    initial_positions = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names])
    plan, log = collections.deque(), collections.defaultdict(list)
    first_lift, lift_t, inference = None, None, 0
    started = time.monotonic()
    save_npz(destination.with_name(destination.stem + "_initial.npz"), {
        "initial_state": np.asarray(initial_state), "sim_state": np.asarray(inner.sim.get_state().flatten()),
        "initial_object_positions": initial_positions,
    })
    diagnostics = []
    for t in range(config["max_steps"]):
        if not plan:
            observation = policy_observation(obs, prompt)
            noise_seed = seed_for(config["noise_base"], config["pair_key"], config["episode"], inference)
            response = request(client, observation, patch_mode, site, noise_seed)
            log["infer_t"].append(t)
            log["chunk"].append(response["actions"])
            log["noise"].append(response["exp1"]["noise"])
            log["noise_seed"].append(noise_seed)
            log["image"].append(observation["observation/image"])
            log["wrist_image"].append(observation["observation/wrist_image"])
            log["state"].append(observation["observation/state"])
            diagnostics.append(response["exp1"]["diagnostics"])
            plan.extend(response["actions"][:5])
            inference += 1
        action = plan.popleft()
        log["action"].append(action)
        obs, _, _, _ = env.step(np.asarray(action).tolist())
        positions = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names])
        eef = obs["robot0_eef_pos"]
        log["obj_pos"].append(positions)
        log["eef_pos_after"].append(eef.copy())
        log["gripper_qpos_after"].append(obs["robot0_gripper_qpos"].copy())
        log["old_goal_ok"].append(goal_values(inner, config["old_goal"]))
        log["new_goal_ok"].append(goal_values(inner, config["new_goal"]))
        lifted = ((positions[:, 2] - initial_positions[:, 2]) > 0.04) & (
            np.linalg.norm(positions - eef[None, :], axis=1) < 0.12)
        eligible_objects = set(inner.objects_dict)
        candidates = [i for i in np.flatnonzero(lifted) if names[int(i)] in eligible_objects]
        if candidates:
            first_lift, lift_t = names[int(candidates[0])], t + 1
            break
    arrays = {key: np.asarray(value) for key, value in log.items()}
    arrays.update(obj_names=np.asarray(names), initial_object_positions=initial_positions)
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite rollout array: {name}")
    save_npz(destination, arrays)
    write_json(destination.with_suffix(".diagnostics.json"), diagnostics)
    return {
        "condition": mode, "site": site, "prompt": prompt, "first_lift": first_lift, "lift_t": lift_t,
        "selected_original": first_lift == config["old_object"],
        "selected_instructed": first_lift == config["new_object"], "steps": t + 1,
        "n_infer": inference, "seconds": time.monotonic() - started,
        "old_goal_at_stop": goal_reached(log["old_goal_ok"][-1]),
        "new_goal_at_stop": goal_reached(log["new_goal_ok"][-1]),
        "artifact": str(destination.relative_to(config["run_dir"])),
        "initial_observation_sha256": expected_hash,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = output_path(config["run_dir"])
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    if client.get_server_metadata().get("config_sha256") != sha256(args.config):
        raise ValueError("Server/client configuration differs")
    suite = benchmark.get_benchmark_dict()["libero_object_task"]()
    task = suite.get_task(config["task_id"])
    if task.name != config["task_name"]:
        raise ValueError("Task ordering changed")
    bddl = ROOT / config["new_bddl"]
    parsed = robosuite_parse_problem(str(bddl))
    if " ".join(parsed["language_instruction"]) != config["new_prompt"]:
        raise ValueError("Prompt does not match simulator language")
    if parsed["goal_state"] != config["new_goal"]:
        raise ValueError("Goal differs from simulator")
    states = suite.get_task_init_states(config["task_id"])
    initial_state = states[config["episode"]]
    seed = seed_for("simulator", config["task_id"], config["episode"])
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    records = []
    try:
        obs = reset_episode(env, initial_state, seed)
        report = preflight(client, policy_observation(obs, config["new_prompt"]), config, run / "preflight")
        # Positive controls first. Do not spend four more rollouts if they lack a choice contrast.
        site = report["selected_site"] or "pg11"
        for condition in ("baseline_new", "baseline_old"):
            obs = reset_episode(env, initial_state, seed)
            r = rollout(client, env, obs, initial_state, config, condition, site, report["observation_sha256"],
                        run / "rollouts" / (condition + ".npz"))
            records.append(r)
            write_json(run / "results.json", records)
            logging.info("%s first_lift=%s steps=%d", condition, r["first_lift"], r["steps"])
        valid = records[0]["selected_instructed"] and records[1]["selected_original"]
        reason = "complete" if valid and report["selected_site"] else "no_valid_behavioral_or_site_contrast"
        if valid and report["selected_site"]:
            for condition in ("self", "dense", "pooled", "pooled_norm"):
                obs = reset_episode(env, initial_state, seed)
                r = rollout(client, env, obs, initial_state, config, condition, site, report["observation_sha256"],
                            run / "rollouts" / (condition + ".npz"))
                records.append(r)
                write_json(run / "results.json", records)
                logging.info("%s first_lift=%s steps=%d", condition, r["first_lift"], r["steps"])
            with np.load(run / records[0]["artifact"]) as a, np.load(run / records[2]["artifact"]) as b:
                for key in ("action", "eef_pos_after", "obj_pos", "noise_seed"):
                    np.testing.assert_array_equal(a[key], b[key], err_msg="Closed-loop self-patch identity: " + key)
        write_json(run / "status.json", {"status": reason, "n_rollouts": len(records),
                                         "selected_site": report["selected_site"], "positive_controls_valid": bool(valid)})
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
