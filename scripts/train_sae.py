#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from drvla.layers import parse_layer
from drvla.train import TrainConfig, load_layer_activations, train_sae

PAPER_K = {"paligemma": 100, "action_expert": 64}


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", required=True, nargs="+", help="One or more activation directories.")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--out", required=True, type=Path, help="SAE directory; the SAE goes to <out>/<layer>/sae.pt.")
    parser.add_argument("--expansion-ratio", type=float, default=defaults.expansion_ratio)
    parser.add_argument("--k", type=int, help="Active features per sample (default: 100 for PaliGemma, 64 for the action expert).")
    parser.add_argument("--auxk", type=int, default=defaults.auxk)
    parser.add_argument("--auxk-coef", type=float, default=defaults.auxk_coef)
    parser.add_argument("--dead-steps-threshold", type=int, default=defaults.dead_steps_threshold)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--warmup-fraction", type=float, default=defaults.warmup_fraction)
    parser.add_argument("--geometric-median-samples", type=int, default=defaults.geometric_median_samples)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    component, _ = parse_layer(args.layer)
    k = args.k if args.k is not None else PAPER_K[component]
    logging.info("k=%d%s", k, "" if args.k is not None else f" (paper setting for {component} layers)")
    config = TrainConfig(
        expansion_ratio=args.expansion_ratio,
        k=k,
        auxk=args.auxk,
        auxk_coef=args.auxk_coef,
        dead_steps_threshold=args.dead_steps_threshold,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_fraction=args.warmup_fraction,
        geometric_median_samples=args.geometric_median_samples,
        seed=args.seed,
        save_every=args.save_every,
    )
    activations = load_layer_activations(args.activations, args.layer)
    logging.info("Loaded %s activations for %s", tuple(activations.shape), args.layer)
    train_sae(activations, config, args.out / args.layer, device=args.device)


if __name__ == "__main__":
    main()
