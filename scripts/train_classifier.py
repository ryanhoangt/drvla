#!/usr/bin/env python3

import argparse
from pathlib import Path

from drvla.classifier import labeled_metric_table, load_labels, train_classifier
from drvla.index import FeatureIndex
from drvla.metrics import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, nargs="+", type=Path, help="One or more label files.")
    parser.add_argument("--index", type=Path, help="Feature index to read metrics from.")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    labels = [entry for path in args.labels for entry in load_labels(path)["labels"]]
    metric_tables = None
    if args.index is not None:
        index = FeatureIndex(args.index)
        metric_tables = {layer: index.metrics(layer) for layer in {entry["layer"] for entry in labels}}
    table = labeled_metric_table(labels, metric_tables)

    description = f"Trained on {', '.join(p.name for p in args.labels)}"
    clf = train_classifier(table, table["y"].to_numpy(), metrics=args.metrics, C=args.C, description=description)
    clf.save(args.out)

    print(f"{clf.n_labeled} labeled features ({int(table['y'].sum())} general)")
    print(f"intercept  {clf.intercept:+.3f}")
    for metric in clf.metrics:
        print(f"{metric:27s}{clf.coefficients[metric]:+.3f}")
    print(f"leave-one-out accuracy: {clf.loo_accuracy:.1%}" if clf.loo_accuracy is not None else "leave-one-out: n/a")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
