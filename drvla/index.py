import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from drvla.layers import parse_layer
from drvla.metrics import MetricAccumulator
from drvla.sae import TopKSAE, encode_batched, load_sae, sae_path
from drvla.store import ActivationStore

logger = logging.getLogger(__name__)

INDEX_FILE = "index.json"


def _merge_top(values, ids, new_values, new_ids, k):
    """Keep the k largest entries per row of ``values`` + ``new_values`` (ids follow along)."""
    values = np.concatenate([values, new_values], axis=1)
    ids = [np.concatenate([a, b], axis=1) for a, b in zip(ids, new_ids)]
    keep = np.argpartition(-values, k - 1, axis=1)[:, :k]
    return np.take_along_axis(values, keep, axis=1), [np.take_along_axis(a, keep, axis=1) for a in ids]


def build_layer_index(
    store: ActivationStore,
    sae: TopKSAE,
    layer: str,
    top_k: int = 100,
    num_top_episodes: int = 10,
    timesteps_per_episode: int = 10,
) -> dict[str, np.ndarray]:
    """Encode every episode of ``layer`` and return the index arrays.

    Returned arrays (F = number of features); missing entries have value -inf and id -1:

    * ``top_values``, ``top_episode_ids``, ``top_timesteps``: (F, top_k), sorted by value.
    * ``episode_ids``, ``episode_max``: (F, num_top_episodes), sorted by episode maximum.
    * ``episode_timesteps``, ``episode_timestep_values``: (F, num_top_episodes, timesteps_per_episode).
    * one (F,) array per column of ``MetricAccumulator.result()``.
    """
    num_features = sae.config.num_features
    E, S = num_top_episodes, timesteps_per_episode
    top_values = np.full((num_features, top_k), -np.inf, dtype=np.float32)
    top_ids = [np.full((num_features, top_k), -1, dtype=np.int32) for _ in range(2)]
    ep_max = np.full((num_features, E), -np.inf, dtype=np.float32)
    ep_ids = np.full((num_features, E), -1, dtype=np.int32)
    ep_ts = np.full((num_features, E, S), -1, dtype=np.int32)
    ep_ts_values = np.full((num_features, E, S), -np.inf, dtype=np.float32)
    accumulator = MetricAccumulator(num_features)

    for episode_id in tqdm(store.episode_ids, desc=f"Indexing {layer}"):
        features = encode_batched(sae, store.activations(layer, episode_id)).numpy()
        accumulator.update(features)
        num_steps = features.shape[0]
        masked = np.where(features > 0, features, -np.inf)

        # Global top-k timesteps.
        k_e = min(top_k, num_steps)
        cand_t = np.argpartition(-masked, k_e - 1, axis=0)[:k_e].T  # (F, k_e)
        cand_v = np.take_along_axis(masked.T, cand_t, axis=1)
        cand_ep = np.full_like(cand_t, episode_id)
        top_values, top_ids = _merge_top(top_values, top_ids, cand_v, [cand_ep, cand_t.astype(np.int32)], top_k)

        # Top episodes by per-episode maximum, keeping each episode's highest timesteps.
        s_e = min(S, num_steps)
        order = np.argsort(-masked, axis=0, kind="stable")[:s_e].T  # (F, s_e)
        ts = np.full((num_features, S), -1, dtype=np.int32)
        ts_values = np.full((num_features, S), -np.inf, dtype=np.float32)
        ts_values[:, :s_e] = np.take_along_axis(masked.T, order, axis=1)
        ts[:, :s_e] = np.where(np.isfinite(ts_values[:, :s_e]), order, -1)
        merged_max = np.concatenate([ep_max, ts_values[:, :1]], axis=1)
        merged_ids = np.concatenate([ep_ids, np.full((num_features, 1), episode_id, np.int32)], axis=1)
        keep = np.argpartition(-merged_max, E - 1, axis=1)[:, :E]
        ep_max = np.take_along_axis(merged_max, keep, axis=1)
        ep_ids = np.take_along_axis(merged_ids, keep, axis=1)
        ep_ts = np.take_along_axis(np.concatenate([ep_ts, ts[:, None]], 1), keep[:, :, None], 1)
        ep_ts_values = np.take_along_axis(np.concatenate([ep_ts_values, ts_values[:, None]], 1), keep[:, :, None], 1)

    order = np.argsort(-top_values, axis=1, kind="stable")
    top_values = np.take_along_axis(top_values, order, axis=1)
    top_ids = [np.take_along_axis(a, order, axis=1) for a in top_ids]
    top_ids = [np.where(np.isfinite(top_values), a, -1) for a in top_ids]

    order = np.argsort(-ep_max, axis=1, kind="stable")
    ep_max = np.take_along_axis(ep_max, order, axis=1)
    ep_ids = np.where(np.isfinite(ep_max), np.take_along_axis(ep_ids, order, axis=1), -1)
    ep_ts = np.take_along_axis(ep_ts, order[:, :, None], axis=1)
    ep_ts_values = np.take_along_axis(ep_ts_values, order[:, :, None], axis=1)

    arrays = {
        "top_values": top_values,
        "top_episode_ids": top_ids[0],
        "top_timesteps": top_ids[1],
        "episode_ids": ep_ids,
        "episode_max": ep_max,
        "episode_timesteps": ep_ts,
        "episode_timestep_values": ep_ts_values,
    }
    for column, values in accumulator.result().items():
        arrays[f"metric_{column}"] = values.to_numpy()
    arrays["num_timesteps"] = np.asarray(accumulator.num_timesteps)
    return arrays


def build_index(
    activation_dir: str | Path,
    sae_dir: str | Path,
    index_dir: str | Path,
    layers: list[str] | None = None,
    top_k: int = 100,
    device: str = "cuda",
):
    store = ActivationStore(activation_dir)
    sae_dir = Path(sae_dir)
    if layers is None:
        layers = [layer for layer in store.layers if sae_path(sae_dir, layer).exists()]
        if not layers:
            raise FileNotFoundError(f"No SAE in {sae_dir} matches a collected layer ({store.layers}).")
        logger.info("Indexing the collected layers that have an SAE: %s", layers)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    info_path = index_dir / INDEX_FILE
    info = json.loads(info_path.read_text()) if info_path.exists() else {"layers": {}}
    for key, value in [("activation_dir", str(Path(activation_dir).resolve())), ("sae_dir", str(sae_dir.resolve()))]:
        if info.get(key, value) != value:
            raise ValueError(f"{info_path} was built from {key}={info[key]}, not {value}. Use a new index directory.")
        info[key] = value

    for layer in layers:
        sae = load_sae(sae_dir, layer, device=device)
        arrays = build_layer_index(store, sae, layer, top_k=top_k)
        # Write-then-rename so readers never see a partially written file.
        np.savez(index_dir / f"{layer}.tmp.npz", **arrays)
        (index_dir / f"{layer}.tmp.npz").replace(index_dir / f"{layer}.npz")
        info["layers"][layer] = {
            "num_features": sae.config.num_features,
            "num_episodes": len(store),
            "num_timesteps": int(arrays["num_timesteps"]),
            "top_k": top_k,
        }
        info_path.with_suffix(".tmp").write_text(json.dumps(info, indent=2))
        info_path.with_suffix(".tmp").replace(info_path)
        logger.info("Indexed %s: %d features over %d episodes", layer, sae.config.num_features, len(store))
        del sae
        if device != "cpu":
            torch.cuda.empty_cache()


class FeatureIndex:
    """Read access to an index built by :func:`build_index`."""

    def __init__(self, index_dir: str | Path):
        self.root = Path(index_dir)
        info_path = self.root / INDEX_FILE
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found; build it with scripts/build_index.py.")
        self.info = json.loads(info_path.read_text())
        self.layers: list[str] = sorted(self.info["layers"], key=lambda l: (parse_layer(l)[0] != "paligemma", parse_layer(l)[1]))
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _arrays(self, layer: str) -> dict[str, np.ndarray]:
        if layer not in self.info["layers"]:
            raise KeyError(f"Layer {layer} is not indexed in {self.root} (have {self.layers}).")
        if layer not in self._cache:
            with np.load(self.root / f"{layer}.npz") as data:
                self._cache[layer] = {key: data[key] for key in data.files}
        return self._cache[layer]

    def metrics(self, layer: str) -> pd.DataFrame:
        """One row per feature with the generality metrics."""
        arrays = self._arrays(layer)
        return pd.DataFrame({key[len("metric_"):]: value for key, value in arrays.items() if key.startswith("metric_")})

    def top_activations(self, layer: str, feature_id: int, n: int | None = None) -> pd.DataFrame:
        arrays = self._arrays(layer)
        valid = arrays["top_episode_ids"][feature_id] >= 0
        table = pd.DataFrame(
            {
                "episode_id": arrays["top_episode_ids"][feature_id][valid],
                "timestep": arrays["top_timesteps"][feature_id][valid],
                "activation": arrays["top_values"][feature_id][valid],
            }
        )
        return table.head(n) if n is not None else table

    def top_episodes(self, layer: str, feature_id: int) -> list[dict]:
        arrays = self._arrays(layer)
        episodes = []
        for rank, episode_id in enumerate(arrays["episode_ids"][feature_id]):
            if episode_id < 0:
                break
            ts = arrays["episode_timesteps"][feature_id, rank]
            valid = ts >= 0
            episodes.append(
                {
                    "episode_id": int(episode_id),
                    "max_activation": float(arrays["episode_max"][feature_id, rank]),
                    "timesteps": ts[valid].tolist(),
                    "activations": arrays["episode_timestep_values"][feature_id, rank][valid].tolist(),
                }
            )
        return episodes
