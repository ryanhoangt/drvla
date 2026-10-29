"""Run exactly one small Exp1 core pilot inside a Slurm MIG allocation."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

from experiments.exp1.common import ROOT, output_path, protected_inventory, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "mig":
        raise ValueError("Exp1 must run in a Slurm mig allocation, not the current CPU session")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one allocated MIG CUDA device is required")
    gpu = torch.cuda.get_device_properties(0)
    memory_gib = gpu.total_memory / 1024**3
    if not 35 <= memory_gib <= 43:
        raise ValueError(f"Expected the requested 40 GB MIG, got {gpu.name}: {memory_gib:.1f} GiB")
    run = output_path(args.run_dir)
    if run.exists():
        raise ValueError("Use a fresh job directory; no automatic resume or overwrite of completed experiments")
    run.mkdir(parents=True)
    logs, temporary, snapshot = run / "logs", run / "tmp", run / "source"
    for directory in (logs, temporary, snapshot):
        directory.mkdir()
    from experiments.rq1.task_meta import BDDL_ROOT, parse

    task_name = "pick_up_the_cream_cheese_and_place_it_in_the_basket"
    old_bddl = BDDL_ROOT / "libero_object" / (task_name + ".bddl")
    new_bddl = BDDL_ROOT / "libero_object_task" / (task_name + ".bddl")
    old, new = parse(old_bddl), parse(new_bddl)
    old_goal = [[p[0].lower(), *p[1:]] for p in old["goal"]]
    new_goal = [[p[0].lower(), *p[1:]] for p in new["goal"]]
    config = {
        "protocol": "exp1-core-v1", "run_dir": str(run), "task_id": 1, "task_name": task_name,
        "pair_key": "libero_object/" + task_name, "episode": 0, "noise_base": 7,
        "old_prompt": " ".join(old["language"].lower().split()),
        "new_prompt": " ".join(new["language"].lower().split()), "old_goal": old_goal, "new_goal": new_goal,
        "old_object": old_goal[0][1], "new_object": new_goal[0][1],
        "old_bddl": str(old_bddl.relative_to(ROOT)), "new_bddl": str(new_bddl.relative_to(ROOT)),
        "max_steps": 180, "replan_steps": 5, "min_dense_projection": 0.05,
        "patch_until": "first lift or 180 actions; all aligned AE denoising calls",
        "horizon_rationale": "120 truncated the old-prompt baseline; prior RQ2 first lift was step 133. "
                             "180 fixed before intervention rollouts; no horizon sweep.",
        "max_rollouts": 6, "primary_endpoint": "first lifted object, NOT task success rate",
    }
    write_json(run / "config.json", config)
    source_hashes = {}
    for path in sorted((ROOT / "experiments/exp1").iterdir()):
        if path.suffix in (".py", ".slurm", ".md"):
            source_hashes[path.name] = sha256(path)
            shutil.copyfile(path, snapshot / path.name)
    write_json(run / "manifest.json", {
        "config_sha256": sha256(run / "config.json"), "source_sha256": source_hashes,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (old_bddl, new_bddl)},
        "resources": {"job_id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
                      "hostname": socket.gethostname(), "gpu_name": gpu.name, "gpu_memory_gib": memory_gib,
                      "cpu_affinity": sorted(os.sched_getaffinity(0)),
                      "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
                      "slurm_memory_mb": os.environ.get("SLURM_MEM_PER_NODE"),
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "torch_version": torch.__version__},
    })
    before = protected_inventory()
    write_json(run / "protected_inventory_before.json", before)
    base_env = dict(os.environ, TMPDIR=str(temporary), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                    JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    # Never overwrite CUDA_VISIBLE_DEVICES: Slurm assigned a specific MIG device.
    server_env = dict(base_env, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    libero_config = run / "libero_config"
    libero_config.mkdir()
    libero_root = ROOT / "openpi/third_party/LIBERO-PRO/libero/libero"
    write_json(libero_config / "config.yaml", {
        "benchmark_root": str(libero_root), "bddl_files": str(libero_root / "bddl_files"),
        "init_states": str(libero_root / "init_files"), "assets": str(libero_root / "assets"),
        "datasets": str(ROOT / "droid_2k"),
    })
    client_env = dict(base_env, LIBERO_CONFIG_PATH=str(libero_config), MUJOCO_GL="osmesa",
                      PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                      PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "openpi/third_party/LIBERO-PRO"),
                                                str(ROOT / "openpi/packages/openpi-client/src")]))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    owned = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Stopping owned Exp1 processes on signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    policy_python = str(ROOT / "openpi/.venv/bin/python")
    libero_python = "/mnt/data/vhoangth2/miniconda3/envs/libero/bin/python"
    try:
        with (logs / "policy.log").open("w") as policy_log, (logs / "simulator.log").open("w") as simulator_log:
            server = subprocess.Popen([policy_python, "-m", "experiments.exp1.serve_policy", "--config",
                                       str(run / "config.json"), "--port", str(port)], cwd=ROOT, env=server_env,
                                      stdout=policy_log, stderr=subprocess.STDOUT)
            owned.append(server)
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server exited {server.returncode}; inspect {logs / 'policy.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Policy startup timed out")
                    time.sleep(1)
            print(f"MIG policy ready on {socket.gethostname()}, port {port}; ONE core pilot", flush=True)
            client = subprocess.Popen([libero_python, "-m", "experiments.exp1.rollout", "--config",
                                       str(run / "config.json"), "--port", str(port)], cwd=ROOT, env=client_env,
                                      stdout=simulator_log, stderr=subprocess.STDOUT)
            owned.append(client)
            if client.wait() != 0:
                raise RuntimeError(f"Core pilot failed; inspect {logs / 'simulator.log'}")
        subprocess.run([policy_python, "-m", "experiments.exp1.analyze", "--run-dir", str(run)],
                       cwd=ROOT, env=base_env, check=True)
        print(f"Done. Read ONE report: {run / 'EXP1_report.md'}", flush=True)
    finally:
        for process in reversed(owned):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        after = protected_inventory()
        changed = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
        write_json(run / "protected_artifacts_audit.json", {
            "checked_files": len(before), "unchanged": not changed, "changed": changed,
        })
        if changed:
            raise RuntimeError(f"Existing RQ1/RQ2 artifacts changed: {changed[:10]}")


if __name__ == "__main__":
    main()
