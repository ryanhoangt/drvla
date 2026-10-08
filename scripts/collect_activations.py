#!/usr/bin/env python3
"""Collect mean-pooled Pi0.5 residual-stream activations for a dataset.

Writes one (T, d) float32 array per episode and layer, the episode's task and
length, and (unless --no-frames) JPEG camera frames for the dashboard. Episodes
that are already complete in --out are skipped, so the script can be resumed
and several shards (--shard / --num-shards) can write to the same directory.

Examples:
    # LIBERO, all 1,693 episodes, the eight layers from the paper
    python scripts/collect_activations.py --dataset libero \
        --checkpoint $CKPT/pi05_libero_pytorch --out activations/pi05_libero

    # DROID, the paper's 2,000-episode subset, shard 3 of 20
    python scripts/collect_activations.py --dataset droid \
        --checkpoint $CKPT/pi05_droid_pytorch --out activations/pi05_droid \
        --droid-rlds-dir /data/droid/1.0.1 --shard 3 --num-shards 20
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from drvla.datasets import make_reader
from drvla.layers import PAPER_LAYERS
from drvla.store import ActivationWriter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSET_IDS = {"libero": "physical-intelligence/libero", "droid": "droid"}

logger = logging.getLogger("collect_activations")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["libero", "droid"])
    parser.add_argument("--checkpoint", required=True, type=Path, help="Pi0.5 PyTorch checkpoint directory.")
    parser.add_argument("--out", required=True, type=Path, help="Activation directory to write.")
    parser.add_argument(
        "--asset-id",
        help="Sub-directory of <checkpoint>/assets with norm_stats.json "
        f"(default: {DEFAULT_ASSET_IDS['libero']} for libero, {DEFAULT_ASSET_IDS['droid']} for droid).",
    )
    parser.add_argument("--layers", nargs="+", default=PAPER_LAYERS)
    parser.add_argument("--max-episodes", type=int, help="Only the first N episodes (for quick tests).")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16, help="Timesteps per forward pass.")
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the flow-matching noise.")
    parser.add_argument(
        "--openpi-server-inputs",
        action="store_true",
        help="Feed the model what openpi's pi05_libero policy server feeds it: no discretized state in the "
        "prompt and a black image in the masked right-wrist slot.",
    )
    parser.add_argument("--device", default="cuda")
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument("--no-frames", action="store_true", help="Do not store camera frames.")
    frames.add_argument("--frames-only", action="store_true", help="Store episode metadata and frames without running the model.")
    parser.add_argument("--frame-size", type=int, default=224, help="Longer side of stored frames, in pixels.")
    parser.add_argument("--libero-repo-id", default="physical-intelligence/libero")
    parser.add_argument("--droid-rlds-dir", help="Directory containing droid/1.0.1 (RLDS).")
    parser.add_argument("--droid-episodes", default=str(REPO_ROOT / "data" / "droid_2k_episodes.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"--shard {args.shard} must be in [0, --num-shards {args.num_shards}).")
    asset_id = args.asset_id if args.asset_id is not None else DEFAULT_ASSET_IDS[args.dataset]

    reader = make_reader(args.dataset, args.libero_repo_id, args.droid_rlds_dir, args.droid_episodes)
    episode_ids = reader.episode_ids[: args.max_episodes] if args.max_episodes else reader.episode_ids
    per_shard = -(-len(episode_ids) // args.num_shards)
    episode_ids = episode_ids[args.shard * per_shard : (args.shard + 1) * per_shard]

    collection = {
        "dataset": args.dataset,
        "source": args.libero_repo_id if args.dataset == "libero" else Path(args.droid_episodes).name,
        "checkpoint": args.checkpoint.resolve().name,
        "asset_id": asset_id,
        "layers": list(args.layers),
        "pooling": "mean over token positions",
        "num_denoising_steps": args.num_denoising_steps,
        "seed": args.seed,
    }
    if args.openpi_server_inputs:
        collection["inputs"] = "openpi policy server (no state in prompt, black masked image)"
    writer = ActivationWriter(args.out, collection, frame_max_side=args.frame_size)
    need_activations, need_frames = not args.frames_only, not args.no_frames
    todo = [e for e in episode_ids if not writer.is_done(e, need_activations, need_frames)]
    logger.info("Shard %d/%d: %d episodes, %d left to do", args.shard, args.num_shards, len(episode_ids), len(todo))
    if not todo:
        return

    extractor = None
    if need_activations:
        from drvla.pi05 import Pi05ActivationExtractor

        extractor = Pi05ActivationExtractor(
            args.checkpoint, asset_id, args.layers, device=args.device,
            num_denoising_steps=args.num_denoising_steps, seed=args.seed,
            discrete_state_input=not args.openpi_server_inputs,
            masked_image_value=-1.0 if args.openpi_server_inputs else 0.0,
        )

    start = time.time()
    for episode in tqdm(reader.iterate(todo), total=len(todo), desc="Episodes"):
        if extractor is not None:
            batches = []
            for i in range(0, episode.length, args.batch_size):
                batch = slice(i, i + args.batch_size)
                batches.append(extractor(episode.images[batch], episode.wrist_images[batch], episode.states[batch], episode.task))
            writer.write_activations(
                episode.episode_id, {layer: np.concatenate([b[layer] for b in batches]) for layer in args.layers}
            )
        if need_frames:
            writer.write_frames(episode.episode_id, episode.images, episode.wrist_images)
        writer.write_episode_meta(episode.episode_id, episode.task, episode.length, episode.source)
    logger.info("Done: %d episodes in %.1f min", len(todo), (time.time() - start) / 60)


if __name__ == "__main__":
    main()
