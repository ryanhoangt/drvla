"""Scoring artifact schema shared by the producer and report reader."""

import numpy as np


def save_scores(path, values, examples, attribution, sensitivity, score, joint_score):
    arrays = {key: value for key, value in values.items() if key != "sensitivity"}
    arrays.update(sample_sensitivity=values["sensitivity"], sensitivity=sensitivity,
                  attribution=attribution, score=score, joint_score=joint_score,
                  task=np.array([r["task"] for r in examples]),
                  episode=np.array([r["episode"] for r in examples]),
                  frame=np.array([r["frame"] for r in examples]), schema_version=np.array(1))
    validate_scores(arrays)
    np.savez_compressed(path, **arrays)


def validate_scores(arrays):
    shape = arrays["f"].shape
    if len(shape) != 2:
        raise ValueError("Expected scoring samples x features")
    samples, features = shape
    for name in ("cf", "effect", "sample_sensitivity"):
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have sample shape {shape}")
    for name in ("attribution", "sensitivity", "score", "joint_score"):
        if arrays[name].shape != (features,):
            raise ValueError(f"{name} must have one summary value per feature")
    if arrays["pooled"].ndim != 2 or arrays["pooled"].shape[0] != samples:
        raise ValueError("Pooled activations must align with scoring examples")
    for name in ("task", "episode", "frame"):
        if arrays[name].shape != (samples,):
            raise ValueError(f"{name} must align with scoring examples")
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite scoring artifact: {name}")


def load_scores(path):
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_scores(arrays)
    return arrays
