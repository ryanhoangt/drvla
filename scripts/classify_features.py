#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from drvla.classifier import GeneralityClassifier
from drvla.index import FeatureIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    index = FeatureIndex(args.index)
    clf = GeneralityClassifier.load(args.classifier)
    tables = []
    for layer in args.layers or index.layers:
        table = index.metrics(layer)
        table = table[table["num_active_episodes"] > 0].copy()
        table["p_general"] = clf.predict_proba(table)
        table.insert(0, "layer", layer)
        tables.append(table)
        n_general = int((table["p_general"] >= 0.5).sum())
        print(f"{layer:32s} {len(table):6d} active features  {n_general:5d} general ({n_general / len(table):.2%})")
    if args.csv is not None:
        pd.concat(tables).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
