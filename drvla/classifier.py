
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from drvla.metrics import METRICS

LABELS = {"general": 1, "memorized": 0}


@dataclass
class GeneralityClassifier:
    metrics: list[str]
    coefficients: dict[str, float]
    intercept: float
    n_labeled: int = 0
    loo_accuracy: float | None = None
    description: str = ""
    extra: dict = field(default_factory=dict)

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        """P(general) for each row of a table containing the classifier's metric columns."""
        missing = [m for m in self.metrics if m not in table.columns]
        if missing:
            raise KeyError(f"Metric columns {missing} are missing from the feature table.")
        logits = self.intercept + sum(self.coefficients[m] * table[m].to_numpy(dtype=np.float64) for m in self.metrics)
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GeneralityClassifier":
        data = json.loads(Path(path).read_text())
        clf = cls(**data)
        if set(clf.coefficients) != set(clf.metrics):
            raise ValueError(f"{path}: coefficients {sorted(clf.coefficients)} do not match metrics {clf.metrics}.")
        return clf


def train_classifier(
    metric_table: pd.DataFrame,
    labels: np.ndarray,
    metrics: list[str] | tuple[str, ...] = METRICS,
    C: float = 1.0,
    description: str = "",
) -> GeneralityClassifier:
    """Fit an L2-regularized logistic regression and report leave-one-out accuracy.

    ``metric_table`` has one row per labeled feature; ``labels`` holds 1 (general)
    or 0 (memorized) for each row.
    """
    from sklearn.linear_model import LogisticRegression

    metrics = list(metrics)
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; choose from {list(METRICS)}.")
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(metric_table):
        raise ValueError(f"{len(labels)} labels for {len(metric_table)} rows.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Need at least one general (1) and one memorized (0) example.")

    X = metric_table[metrics].to_numpy(dtype=np.float64)

    def fit(X_fit, y_fit):
        return LogisticRegression(C=C, max_iter=1000, random_state=42).fit(X_fit, y_fit)

    model = fit(X, labels)
    correct, evaluated = 0, 0
    for i in range(len(labels)):
        keep = np.arange(len(labels)) != i
        if len(np.unique(labels[keep])) < 2:
            continue
        correct += int(fit(X[keep], labels[keep]).predict(X[i : i + 1])[0] == labels[i])
        evaluated += 1

    return GeneralityClassifier(
        metrics=metrics,
        coefficients={m: float(c) for m, c in zip(metrics, model.coef_[0])},
        intercept=float(model.intercept_[0]),
        n_labeled=len(labels),
        loo_accuracy=correct / evaluated if evaluated else None,
        description=description,
        extra={"C": C},
    )


def load_labels(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    for entry in data["labels"]:
        if entry["label"] not in LABELS:
            raise ValueError(f"{path}: label {entry['label']!r} must be one of {list(LABELS)}.")
    return data


def save_labels(path: str | Path, labels: list[dict], description: str = ""):
    for entry in labels:
        if entry["label"] not in LABELS:
            raise ValueError(f"Label {entry['label']!r} must be one of {list(LABELS)}.")
    Path(path).write_text(json.dumps({"description": description, "labels": labels}, indent=2))


def labeled_metric_table(labels: list[dict], metric_tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per label with its metrics and a 0/1 ``y`` column.

    Metrics come from ``metric_tables[layer]`` (e.g. a feature index) when given,
    otherwise from the ``metrics`` stored in each label entry.
    """
    rows = []
    for entry in labels:
        if metric_tables is not None:
            table = metric_tables.get(entry["layer"])
            if table is None:
                raise KeyError(f"No metrics for layer {entry['layer']}.")
            match = table[table["feature_id"] == entry["feature_id"]]
            if match.empty:
                raise KeyError(f"Feature {entry['feature_id']} not found in layer {entry['layer']}.")
            metrics = {m: float(match.iloc[0][m]) for m in METRICS}
        else:
            if "metrics" not in entry:
                raise KeyError(
                    f"Label for feature {entry['feature_id']} ({entry['layer']}) has no stored metrics; "
                    "pass a feature index to look them up."
                )
            metrics = entry["metrics"]
        rows.append({"layer": entry["layer"], "feature_id": entry["feature_id"], **metrics, "y": LABELS[entry["label"]]})
    return pd.DataFrame(rows)
