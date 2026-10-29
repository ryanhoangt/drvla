"""Own both policy/simulator processes; all evaluation outputs live under Exp2."""

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time
import urllib.request

from experiments.exp2.common import ROOT, destination, sha256, write_json
from experiments.exp2.arm1.common import protected as inventory
from experiments.exp2.arm1.resume import seed_results


def evaluate(training_run, eval_dir, task_ids, suites, episodes, noise_base, resume_evaluation=None):
    import torch

    if not os.getenv("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Evaluation must run in its assigned Slurm GPU allocation")
    status = json.loads((training_run / "status.json").read_text())
    if status["status"] != "training_complete":
        raise ValueError("Training did not complete")
    audit = json.loads((training_run / "protected_artifacts_audit.json").read_text())
    if not audit["unchanged"]:
        raise ValueError("Training modified protected scientific inputs")
    eval_dir = destination(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("logs", "tmp", "libero_config"):
        (eval_dir / folder).mkdir()
    data_path = Path(json.loads((training_run / "manifest.json").read_text())["arguments"]["data_dir"])
    data = json.loads((data_path / "manifest.json").read_text())
    config = {"protocol": "exp2-arm1-v1", "training_run": str(training_run), "eval_dir": str(eval_dir),
              "task_ids": task_ids, "suites": suites, "episodes": episodes, "noise_base": noise_base,
              "arms": ["sft_pg_only_off", "acttask_off", "acttask_on", "random_acttask_off", "random_acttask_on"],
              "training_prompts": data["prompts"], "horizon": 280, "replan_steps": 5,
              "projection_removed": False, "explicit_hook_modes": True,
              "primary": "new-goal completion, correct first lifted object, trap events and trap+failure separately",
              "probe_caveat": "Frozen ID linear probes measure decodability; drift can change probabilities "
                              "without changing instruction use. Check each checkpoint's ID probe accuracy.",
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                                for p in (ROOT / "experiments/exp2/arm1").glob("*.py")}}
    if resume_evaluation:
        config["resume_evaluation"] = str(Path(resume_evaluation).resolve())
        reused = seed_results(config, resume_evaluation)
        print(f"Resuming evaluation: {reused} committed rollouts reused; no training.", flush=True)
    write_json(eval_dir / "config.json", config)
    source = eval_dir / "source"
    source.mkdir()
    for path in (ROOT / "experiments/exp2/arm1").glob("*.py"):
        shutil.copyfile(path, source / path.name)
    libero_root = ROOT / "openpi/third_party/LIBERO-PRO/libero/libero"
    write_json(eval_dir / "libero_config/config.yaml", {
        "benchmark_root": str(libero_root), "bddl_files": str(libero_root / "bddl_files"),
        "init_states": str(libero_root / "init_files"), "assets": str(libero_root / "assets"),
        "datasets": str(ROOT / "droid_2k"),
    })
    base_env = dict(os.environ, TMPDIR=str(eval_dir / "tmp"), PYTHONDONTWRITEBYTECODE="1",
                    PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    server_env = dict(base_env, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    client_env = dict(base_env, LIBERO_CONFIG_PATH=str(eval_dir / "libero_config"), MUJOCO_GL="osmesa",
                      PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                      PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "openpi/third_party/LIBERO-PRO"),
                                                str(ROOT / "openpi/packages/openpi-client/src")]))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    owned = []
    def protected_inputs():
        values = inventory()
        # Protect the trained models, prior evaluation and reports, too.
        for folder in {training_run, Path(resume_evaluation).resolve() if resume_evaluation else training_run}:
            for path in folder.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and eval_dir not in path.parents:
                    stat = path.stat()
                    values[str(path)] = [stat.st_size, stat.st_mtime_ns]
        return values

    before = protected_inputs()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Stopping owned Exp2 processes on signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (eval_dir / "logs/policy.log").open("w") as policy_log, (
                eval_dir / "logs/simulator.log").open("w") as simulator_log:
            server = subprocess.Popen([str(ROOT / "openpi/.venv/bin/python"), "-u", "-m",
                                       "experiments.exp2.arm1.serve_policy", "--run-dir", str(training_run),
                                       "--port", str(port)], cwd=ROOT, env=server_env,
                                      stdout=policy_log, stderr=subprocess.STDOUT)
            owned.append(server)
            deadline = time.monotonic()+600
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Policy failed; inspect " + str(eval_dir / "logs/policy.log"))
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy startup timed out")
                    time.sleep(1)
            simulator = subprocess.Popen(["/mnt/data/vhoangth2/miniconda3/envs/libero/bin/python", "-u", "-m",
                                          "experiments.exp2.arm1.rollout", "--config", str(eval_dir / "config.json"),
                                          "--port", str(port)], cwd=ROOT, env=client_env,
                                         stdout=simulator_log, stderr=subprocess.STDOUT)
            owned.append(simulator)
            offset = 0
            while simulator.poll() is None:
                # Forward early behavioral outcomes to the Slurm stdout log.
                with (eval_dir / "logs/simulator.log").open() as stream:
                    stream.seek(offset)
                    for line in stream:
                        if "RESULT " in line or "POSITIVE_CANDIDATE" in line:
                            print(line.strip(), flush=True)
                    offset = stream.tell()
                time.sleep(2)
            if simulator.returncode:
                raise RuntimeError("Simulator failed; inspect " + str(eval_dir / "logs/simulator.log"))
        print(f"Evaluation completed: {eval_dir}", flush=True)
    finally:
        for process in reversed(owned):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        after = protected_inputs()
        changed = [p for p in set(before)|set(after) if before.get(p) != after.get(p)]
        write_json(eval_dir / "protected_artifacts_audit.json", {"checked_files": len(before),
                                                               "unchanged": not changed, "changed": changed})
        if changed:
            raise RuntimeError("Protected artifacts changed: " + str(changed[:10]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--suites", nargs="+", default=["libero_object", "libero_object_task"])
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--noise-base", type=int, default=7)
    parser.add_argument("--resume-evaluation", type=Path)
    args = parser.parse_args()
    evaluate(args.run_dir.resolve(), args.eval_dir, args.task_ids, args.suites, args.episodes, args.noise_base,
             args.resume_evaluation)


if __name__ == "__main__":
    main()
