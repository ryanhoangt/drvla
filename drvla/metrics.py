

import numpy as np
import pandas as pd

ONSET_THRESHOLD = 0.1

METRICS = ("episode_coverage", "mean_onset_count", "mean_activation_magnitude", "relative_run_length")

METRIC_LABELS = {
    "episode_coverage": "Episode coverage (c)",
    "mean_onset_count": "Mean onset count (o)",
    "mean_activation_magnitude": "Mean activation magnitude (a)",
    "relative_run_length": "Relative run length (l_r)",
}


def episode_onsets(features: np.ndarray, threshold: float = ONSET_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Onset counts and number of 'on' timesteps per feature for one (T, F) episode."""
    inactive = np.ones(features.shape[1], dtype=bool)
    onsets = np.zeros(features.shape[1], dtype=np.int64)
    on_steps = np.zeros(features.shape[1], dtype=np.int64)
    above = features > threshold
    zero = features == 0.0
    for t in range(features.shape[0]):
        fires = inactive & above[t]
        onsets += fires
        inactive[fires] = False
        inactive[zero[t]] = True
        on_steps += ~inactive
    return onsets, on_steps


class MetricAccumulator:
    """Accumulates the generality metrics over episodes, one (T, F) array at a time."""

    def __init__(self, num_features: int, onset_threshold: float = ONSET_THRESHOLD):
        self.num_features = num_features
        self.onset_threshold = onset_threshold
        self.num_episodes = 0
        self.num_timesteps = 0
        self.active_episodes = np.zeros(num_features, dtype=np.int64)
        self.active_timesteps = np.zeros(num_features, dtype=np.int64)
        self.max_sum = np.zeros(num_features, dtype=np.float64)
        self.onset_episodes = np.zeros(num_features, dtype=np.int64)
        self.onset_sum = np.zeros(num_features, dtype=np.float64)
        self.relative_run_sum = np.zeros(num_features, dtype=np.float64)
        self.max_activation = np.zeros(num_features, dtype=np.float64)

    def update(self, features: np.ndarray):
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError(f"Expected (T, {self.num_features}) features, got {features.shape}.")
        num_steps = features.shape[0]
        self.num_episodes += 1
        self.num_timesteps += num_steps

        episode_max = features.max(axis=0)
        active = episode_max > 0
        self.active_episodes += active
        self.active_timesteps += (features > 0).sum(axis=0)
        self.max_sum += np.where(active, episode_max, 0.0)
        self.max_activation = np.maximum(self.max_activation, episode_max)

        onsets, on_steps = episode_onsets(features, self.onset_threshold)
        has_onset = onsets > 0
        self.onset_episodes += has_onset
        self.onset_sum += onsets
        run_length = on_steps / np.maximum(onsets, 1)
        self.relative_run_sum += np.where(has_onset, run_length / num_steps, 0.0)

    def result(self) -> pd.DataFrame:
        if self.num_episodes == 0:
            raise ValueError("No episodes were accumulated.")
        return pd.DataFrame(
            {
                "feature_id": np.arange(self.num_features),
                "episode_coverage": self.active_episodes / self.num_episodes,
                "mean_onset_count": self.onset_sum / np.maximum(self.onset_episodes, 1),
                "mean_activation_magnitude": self.max_sum / np.maximum(self.active_episodes, 1),
                "relative_run_length": self.relative_run_sum / np.maximum(self.onset_episodes, 1),
                "num_active_episodes": self.active_episodes,
                "activation_frequency": self.active_timesteps / self.num_timesteps,
                "max_activation": self.max_activation,
            }
        )
