"""Launch an owned RQ2 server and paired simulator workers, then analyze results."""

import argparse
import concurrent.futures
import os
from pathlib import Path
import socket
import shutil
import subprocess
import time
import urllib.request

from experiments.rq2.common import ROOT, manifest, output_path, sha256, write_json


def protected_inventory():
    """Metadata audit of existing scientific artifacts; do not rewrite or hash huge arrays."""
    result = {}
    for directory in ("activations", "saes", "index", "checkpoints", "workspace/rq1", "experiments/rq1"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                st = path.stat()
                result[str(path.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "workspace/rq2/features_pg5_k16_env/features.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--intervene-calls", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2 or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Use one or two workers and unique task IDs on this allocation")
    args.run_dir = output_path(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs = args.run_dir / "logs"
    logs.mkdir(exist_ok=True)
    temporary = args.run_dir / "tmp"
    temporary.mkdir(exist_ok=True)
    policy_python = ROOT / "openpi/.venv/bin/python"
    libero_python = Path("/mnt/data/vhoangth2/miniconda3/envs/libero/bin/python")
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration["source_sha256"] = {
        p.name: sha256(p) for p in (ROOT / "experiments/rq2").glob("*.py")
    }
    configuration["features_sha256"] = sha256(args.features)
    manifest(args.run_dir / "run_manifest.json", configuration)
    snapshot_dir = args.run_dir / "source"
    snapshot_dir.mkdir(exist_ok=True)
    for name, checksum in configuration["source_sha256"].items():
        destination = snapshot_dir / name
        if destination.exists():
            if sha256(destination) != checksum:
                raise ValueError(f"Existing source snapshot differs: {destination}")
        else:
            shutil.copyfile(ROOT / "experiments/rq2" / name, destination)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))  # Refuse to interfere with another server.
    before = protected_inventory()
    write_json(args.run_dir / "protected_inventory_before.json", before)
    server_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false", TMPDIR=str(temporary),
                      PYTHONUNBUFFERED="1")
    # Keep LIBERO's configuration and all new logs within this experiment.
    libero_config = args.run_dir / "libero_config"
    libero_config.mkdir(exist_ok=True)
    libero_root = ROOT / "openpi/third_party/LIBERO-PRO/libero/libero"
    write_json(libero_config / "config.yaml", {
        "benchmark_root": str(libero_root), "bddl_files": str(libero_root / "bddl_files"),
        "init_states": str(libero_root / "init_files"), "datasets": str(ROOT / "droid_2k"),
        "assets": str(libero_root / "assets"),
    })  # JSON is also valid YAML.
    client_env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(ROOT), str(ROOT / "openpi/third_party/LIBERO-PRO"),
        str(ROOT / "openpi/packages/openpi-client/src"),
    ]), MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        LIBERO_CONFIG_PATH=str(libero_config), TMPDIR=str(temporary), PYTHONUNBUFFERED="1")
    server = None
    try:
        with (logs / "server.log").open("a") as server_log:
            server = subprocess.Popen([
                str(policy_python), "-m", "experiments.rq2.serve_policy_interventions",
                "--features", str(args.features), "--port", str(args.port),
            ], cwd=ROOT, env=server_env, stdout=server_log, stderr=subprocess.STDOUT)
            write_json(args.run_dir / "processes.json", {"server_pid": server.pid, "port": args.port})
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; inspect {logs / 'server.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy server did not start within five minutes")
                    time.sleep(1)
            print(f"Server ready pid={server.pid}; launching {args.workers} simulator worker(s)", flush=True)

            def run_task(task_id):
                command = [str(libero_python), "-m", "experiments.rq2.rollout_logger",
                           "--features", str(args.features), "--run-dir", str(args.run_dir),
                           "--tasks", str(task_id), "--port", str(args.port), "--alpha", str(args.alpha),
                           "--intervene-calls", str(args.intervene_calls), "--max-steps", str(args.max_steps),
                           "--episodes", *map(str, args.episodes), "--noise-seeds", *map(str, args.noise_seeds)]
                if args.verify_only:
                    command.append("--verify-only")
                if args.video:
                    command.append("--video")
                with (logs / f"task{task_id:02d}.log").open("a") as log:
                    result = subprocess.run(command, cwd=ROOT, env=client_env, stdout=log, stderr=subprocess.STDOUT)
                print(f"Task {task_id}: worker exit {result.returncode}", flush=True)
                if result.returncode:
                    raise RuntimeError(f"Task {task_id} failed; inspect {logs / f'task{task_id:02d}.log'}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(run_task, args.tasks))
        if not args.verify_only:
            subprocess.run([str(policy_python), "-m", "experiments.rq2.analyze_rq2",
                            "--run-dir", str(args.run_dir)], cwd=ROOT, check=True)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        after = protected_inventory()
        changed = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
        write_json(args.run_dir / "protected_artifacts_audit.json", {
            "checked_files": len(before), "changed": changed, "unchanged": not changed,
        })
        if changed:
            raise RuntimeError(f"Protected artifact metadata changed: {changed[:10]}")


if __name__ == "__main__":
    main()
