"""Build a modest, episode-disjoint ID-only cache, without downloading data."""

import argparse
import io
import json

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from experiments.exp2.common import DATASET, ROOT, destination, sha256, write_json


def prepare(out):
    out = destination(out)
    if out.exists():
        raise FileExistsError(f"Use a fresh data directory: {out}")
    out.mkdir(parents=True)
    tasks = [json.loads(line) for line in (DATASET / "meta/tasks.jsonl").read_text().splitlines()]
    # Names, not historical failure outcomes, define this suite.
    bddls = ROOT / "openpi/third_party/LIBERO-PRO/libero/libero/bddl_files/libero_object"
    names = {p.stem.replace("_", " ") for p in bddls.glob("*.bddl")}
    prompts = sorted(t["task"] for t in tasks if t["task"] in names)
    if len(prompts) != 10:
        raise ValueError(f"Expected exactly ten object tasks: {prompts}")
    episodes = [json.loads(line) for line in (DATASET / "meta/episodes.jsonl").read_text().splitlines()]
    from experiments.rq1.task_meta import parse

    scenes = {p.stem.replace("_", " "): parse(p) for p in bddls.glob("*.bddl")}
    alternatives = {prompt: [q for q in prompts if q != prompt
                            and scenes[q]["goal"][0][1] in scenes[prompt]["init"]]
                    for prompt in prompts}
    if any(len(q) < 2 for q in alternatives.values()):
        raise ValueError("Need two counterfactual training targets present in each ID scene")
    assignments, sources, counts = [], {}, {}
    for task, prompt in enumerate(prompts):
        candidates = sorted(e["episode_index"] for e in episodes if e["tasks"] == [prompt])
        if len(candidates) < 28:
            raise ValueError(f"Not enough ID demonstrations: {prompt}: {len(candidates)}")
        # Four scoring episodes are included in the 24 rectification demos.
        chosen = {"score": candidates[:4], "train": candidates[:24], "validation": candidates[-4:]}
        assert not set(chosen["train"]) & set(chosen["validation"])
        counts[prompt] = chosen
        for split, ids in chosen.items():
            arrays, rows = [], []
            for episode in ids:
                path = DATASET / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
                stat = path.stat()
                sources[str(path)] = [stat.st_size, stat.st_mtime_ns]
                table = pq.read_table(path)
                fractions = (0.0, 0.25) if split == "score" else (
                    (0.0, 0.3, 0.6) if split == "validation" else (0.0, 0.15, 0.3, 0.45, 0.6, 0.75))
                action = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
                for fraction in fractions:
                    frame = min(int(fraction * len(table)), len(table) - 1)
                    row = table.slice(frame, 1).to_pylist()[0]
                    images = []
                    for key in ("image", "wrist_image"):
                        image = Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB")
                        images.append(np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR)))
                    chunk = action[np.minimum(np.arange(frame, frame + 10), len(action) - 1)]
                    arrays.append((images[0], images[1], np.asarray(row["state"], np.float32), chunk))
                    rows.append({"task": task, "prompt": prompt, "episode": episode, "frame": frame})
            file = f"{split}_task{task:02d}.npz"
            np.savez_compressed(out / file, image=np.stack([a[0] for a in arrays]),
                                wrist=np.stack([a[1] for a in arrays]), state=np.stack([a[2] for a in arrays]),
                                actions=np.stack([a[3] for a in arrays]))
            assignments.extend({**row, "split": split, "file": file, "row": i} for i, row in enumerate(rows))
    write_json(out / "manifest.json", {"dataset": str(DATASET), "prompts": prompts,
               "counterfactual_candidates": alternatives,
               "episode_splits": counts, "samples": assignments, "source_metadata": sources,
               "metadata_sha256": {f: sha256(DATASET / "meta" / f)
                                   for f in ("info.json", "episodes.jsonl", "tasks.jsonl")},
               "notes": "Validation episodes held out of Exp2 rectification/scoring, NOT original policy SFT. "
                        "No OOD images, labels, outcomes or prompt mappings used for scoring."})
    print(f"Prepared {len(assignments)} ID-only samples in {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    prepare(parser.parse_args().out)
