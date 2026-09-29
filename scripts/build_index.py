#!/usr/bin/env python3
"""Build the feature index (top activations and generality metrics) used by the dashboard.

Encodes every collected episode with the SAE of each layer. By default indexes
every collected layer that has an SAE in --saes.

Example:
    python scripts/build_index.py --activations activations/pi05_droid \
        --saes saes/pi05_droid --out index/pi05_droid
"""

import argparse
import logging
from pathlib import Path

import torch

from drvla.index import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", required=True, type=Path)
    parser.add_argument("--saes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--top-k", type=int, default=100, help="Top timesteps stored per feature.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_index(args.activations, args.saes, args.out, layers=args.layers, top_k=args.top_k, device=args.device)


if __name__ == "__main__":
    main()
