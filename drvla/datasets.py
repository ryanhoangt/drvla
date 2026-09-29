"""Episode readers for the two datasets used in the paper.

* LIBERO: ``physical-intelligence/libero`` on the Hugging Face Hub, the LeRobot
  dataset that pi05_libero was fine-tuned on (1,693 episodes). Needs ``lerobot``,
  which openpi installs.
* DROID: the v1.0.1 RLDS release, restricted to a list of episodes. The paper's
  2,000-episode subset is ``data/droid_2k_episodes.json``. Needs
  ``tensorflow-datasets`` (openpi's ``rlds`` dependency group).

Both yield :class:`Episode` objects with the two camera streams and the robot
state that Pi0.5 consumes.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass
class Episode:
    episode_id: int
    task: str
    images: np.ndarray  # (T, H, W, 3) uint8, base camera
    wrist_images: np.ndarray  # (T, H, W, 3) uint8, wrist camera
    states: np.ndarray  # (T, state_dim) float32, robot state fed to Pi0.5
    source: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.images)


def _to_uint8_hwc(image) -> np.ndarray:
    """LeRobot (C, H, W) float image in [0, 1] -> (H, W, C) uint8."""
    array = image.permute(1, 2, 0).numpy()
    return (255 * array).astype(np.uint8)


class LiberoEpisodes:
    """Episodes of a LeRobot LIBERO dataset; episode ids are LeRobot ``episode_index`` values."""

    name = "libero"
    default_asset_id = "physical-intelligence/libero"

    def __init__(self, repo_id: str = "physical-intelligence/libero"):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.dataset = LeRobotDataset(repo_id)
        self.episode_ids = list(range(self.dataset.num_episodes))

    def load(self, episode_id: int) -> Episode:
        start = int(self.dataset.episode_data_index["from"][episode_id])
        end = int(self.dataset.episode_data_index["to"][episode_id])
        samples = [self.dataset[i] for i in range(start, end)]
        if int(samples[0]["episode_index"]) != episode_id:
            raise ValueError(f"Frame {start} belongs to episode {int(samples[0]['episode_index'])}, not {episode_id}.")
        return Episode(
            episode_id=episode_id,
            task=samples[0]["task"],
            images=np.stack([_to_uint8_hwc(s["image"]) for s in samples]),
            wrist_images=np.stack([_to_uint8_hwc(s["wrist_image"]) for s in samples]),
            states=np.stack([s["state"].numpy() for s in samples]).astype(np.float32),
            source={"repo_id": self.repo_id, "episode_index": episode_id},
        )

    def iterate(self, episode_ids: Iterable[int]) -> Iterator[Episode]:
        for episode_id in episode_ids:
            yield self.load(episode_id)


class DroidEpisodes:
    """Selected episodes of DROID v1.0.1 (RLDS).

    ``episode_list`` is a JSON file
    ``{"episodes": [{"episode_id", "key", "task", "annotated"}, ...]}`` where
    ``key`` is ``"<recording_folderpath>--<file_path>"`` from the episode metadata
    and ``task`` is the prompt given to the policy. For ``annotated`` episodes the
    first non-empty of ``language_instruction``, ``language_instruction_2`` and
    ``language_instruction_3`` must equal ``task``; episodes without any
    instruction must be listed with ``"annotated": false`` and an explicit prompt.
    The state is the 7 joint positions followed by the gripper position; the
    cameras are ``exterior_image_1_left`` and ``wrist_image_left``.
    """

    name = "droid"
    default_asset_id = "droid"

    def __init__(self, rlds_dir: str | Path, episode_list: str | Path):
        self.rlds_dir = Path(rlds_dir)
        entries = json.loads(Path(episode_list).read_text())["episodes"]
        self.by_key = {entry["key"]: entry for entry in entries}
        if len(self.by_key) != len(entries):
            raise ValueError(f"{episode_list} lists the same key more than once.")
        self.episode_ids = sorted(entry["episode_id"] for entry in entries)

    def _dataset(self):
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")  # keep the GPU for the policy
        # tfds expects <data_dir>/droid/1.0.1/
        data_dir = self.rlds_dir
        if data_dir.name == "1.0.1":
            data_dir = data_dir.parent.parent
        elif data_dir.name == "droid":
            data_dir = data_dir.parent
        builder = tfds.builder("droid", data_dir=str(data_dir), version="1.0.1")
        return builder.as_dataset(split="train", shuffle_files=False).prefetch(1)

    def iterate(self, episode_ids: Iterable[int]) -> Iterator[Episode]:
        """Yield the requested episodes in RLDS order (one pass over the dataset)."""
        wanted = set(episode_ids)
        remaining = {key: entry for key, entry in self.by_key.items() if entry["episode_id"] in wanted}
        if len(remaining) != len(wanted):
            raise KeyError(f"Episode ids {sorted(wanted - {e['episode_id'] for e in remaining.values()})} are not listed.")
        for position, ep in enumerate(self._dataset()):
            if not remaining:
                break
            meta = ep["episode_metadata"]
            key = f"{meta['recording_folderpath'].numpy().decode()}--{meta['file_path'].numpy().decode()}"
            entry = remaining.pop(key, None)
            if entry is None:
                continue
            yield self._read(ep, entry, position)
        if remaining:
            raise KeyError(f"{len(remaining)} listed episodes were not found in {self.rlds_dir}.")

    @staticmethod
    def _read(ep, entry: dict, position: int) -> Episode:
        images, wrist_images, states, instruction = [], [], [], None
        for t, step in enumerate(ep["steps"]):
            if t == 0:
                for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
                    text = step[key].numpy().decode("utf-8")
                    if text.strip():
                        instruction = text
                        break
            obs = step["observation"]
            images.append(obs["exterior_image_1_left"].numpy())
            wrist_images.append(obs["wrist_image_left"].numpy())
            states.append(np.concatenate([obs["joint_position"].numpy(), obs["gripper_position"].numpy().reshape(-1)]))
        if entry["annotated"] and instruction != entry["task"]:
            raise ValueError(f"Episode {entry['key']}: RLDS instruction {instruction!r} != listed task {entry['task']!r}.")
        if not entry["annotated"] and instruction is not None:
            raise ValueError(f"Episode {entry['key']} is listed as unannotated but has instruction {instruction!r}.")
        return Episode(
            episode_id=entry["episode_id"],
            task=entry["task"],
            images=np.stack(images).astype(np.uint8),
            wrist_images=np.stack(wrist_images).astype(np.uint8),
            states=np.stack(states).astype(np.float32),
            source={"key": entry["key"], "rlds_position": position},
        )


def make_reader(dataset: str, libero_repo_id: str, droid_rlds_dir: str | None, droid_episode_list: str | None):
    if dataset == "libero":
        return LiberoEpisodes(libero_repo_id)
    if dataset == "droid":
        if droid_rlds_dir is None or droid_episode_list is None:
            raise ValueError("DROID needs both the RLDS directory and an episode list.")
        return DroidEpisodes(droid_rlds_dir, droid_episode_list)
    raise ValueError(f"Unknown dataset {dataset!r}; expected 'libero' or 'droid'.")
