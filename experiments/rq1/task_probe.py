"""Does the policy's representation follow the scene (original task) or the language (instructed task)?

Trains a 40-way linear probe (softmax regression) on the training activations of each layer to
predict the LIBERO task, then applies it to every inference call of the ``_task`` rollouts and
reports P(original task) and P(instructed task), split by outcome (success / trap / other failure).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from analyze_rq1 import BASE_SUITES, instructed_task, task_language  # noqa: E402
from drvla.store import ActivationStore  # noqa: E402
from task_meta import suite_meta  # noqa: E402


def train_probe(x, y, n_cls, device, epochs=300):
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    X = torch.tensor((x - mu) / sd, device=device)
    Y = torch.tensor(y, device=device)
    W = torch.zeros(x.shape[1], n_cls, device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, Y)
        loss.backward()
        opt.step()
    return lambda z: torch.softmax(torch.tensor((z - mu) / sd, device=device) @ W + b, -1).detach().cpu().numpy()


def main(rollouts, activations, out, layers, device="cuda"):
    rollouts, out = Path(rollouts), Path(out)
    store = ActivationStore(activations)
    tasks = sorted({store.episode(e)["task"] for e in store.episode_ids})
    ids = np.array(store.episode_ids)
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    held = set(ids[: len(ids) // 5].tolist())
    base_metas = {b: suite_meta(b) for b in BASE_SUITES}
    rows = []
    for layer in layers:
        xs, ys, xh, yh = [], [], [], []
        for e in store.episode_ids:
            x = np.asarray(store.activations(layer, e), dtype=np.float32)[::3]
            k = tasks.index(store.episode(e)["task"])
            (xh if e in held else xs).append(x)
            (yh if e in held else ys).extend([k] * len(x))
        probe = train_probe(np.concatenate(xs), np.array(ys), len(tasks), device)
        acc = float((probe(np.concatenate(xh)).argmax(1) == np.array(yh)).mean())
        print(f"{layer}: held-out episode task accuracy {acc:.3f}")
        for suite_dir in sorted(rollouts.glob("libero_*")):
            suite = suite_dir.name
            base = next(b for b in BASE_SUITES if suite == b or suite.startswith(b + "_"))
            metas = suite_meta(suite) if suite != base else base_metas[base]
            for line in (suite_dir / "labels.jsonl").read_text().splitlines():
                r = json.loads(line)
                orig = task_language(r["task"] + ".bddl")
                instr = instructed_task(metas[r["task"] + ".bddl"], base_metas[base]) if suite.endswith("_task") else orig
                d = np.load(suite_dir / f"task{r['task_id']:02d}_ep{r['episode']:03d}.npz")
                p = probe(np.asarray(d["act/" + layer], dtype=np.float32))
                ko = tasks.index(orig)
                ki = tasks.index(instr) if instr in tasks else None
                for i in range(len(p)):
                    rows.append({"layer": layer, "suite": suite, "task_id": r["task_id"], "episode": r["episode"],
                                 "success": r["success"], "trap": r.get("trap"), "i": i, "probe_acc": acc,
                                 "p_orig": float(p[i, ko]), "p_instr": float(p[i, ki]) if ki is not None else np.nan,
                                 "argmax_orig": bool(p[i].argmax() == ko),
                                 "argmax_instr": bool(ki is not None and p[i].argmax() == ki)})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "task_probe.parquet")
    return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
