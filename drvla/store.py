"""On-disk layout of a collected activation dataset.

::

    <root>/
      collection.json                 how the activations were produced (model, dataset, layers)
      episodes/episode_000000.json    {"episode_id", "task", "length", "source"}
      <layer>/episode_000000.npy      (T, d) float32, one mean-pooled vector per timestep
      frames/episode_000000.npz       JPEG-encoded camera frames for the dashboard (optional)

Episode files are written independently, so several collection shards can write
into the same directory at once.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

COLLECTION_FILE = "collection.json"
CAMERAS = ("main", "wrist")


def _episode_name(episode_id: int) -> str:
    return f"episode_{episode_id:06d}"


class ActivationStore:
    """Read access to an activation directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        collection_path = self.root / COLLECTION_FILE
        if not collection_path.exists():
            raise FileNotFoundError(
                f"{collection_path} not found. Is {self.root} an activation directory written by "
                "scripts/collect_activations.py?"
            )
        self.collection = json.loads(collection_path.read_text())
        self.layers: list[str] = self.collection["layers"]

        episode_files = sorted((self.root / "episodes").glob("episode_*.json"))
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata in {self.root / 'episodes'}.")
        self.episodes: dict[int, dict] = {}
        for path in episode_files:
            meta = json.loads(path.read_text())
            self.episodes[meta["episode_id"]] = meta
        self.episode_ids: list[int] = sorted(self.episodes)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def episode(self, episode_id: int) -> dict:
        if episode_id not in self.episodes:
            raise KeyError(f"Episode {episode_id} is not in {self.root}.")
        return self.episodes[episode_id]

    def activations(self, layer: str, episode_id: int) -> np.ndarray:
        if layer not in self.layers:
            raise KeyError(f"Layer {layer} was not collected in {self.root} (have {self.layers}).")
        path = self.root / layer / f"{_episode_name(episode_id)}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing activations {path}.")
        return np.load(path, mmap_mode="r")

    def has_frames(self, episode_id: int) -> bool:
        return (self.root / "frames" / f"{_episode_name(episode_id)}.npz").exists()

    def frames(self, episode_id: int) -> "EpisodeFrames":
        path = self.root / "frames" / f"{_episode_name(episode_id)}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No frames for episode {episode_id} ({path}). Collect with frames enabled, or run "
                "scripts/collect_activations.py --frames-only."
            )
        return EpisodeFrames(path)


class EpisodeFrames:
    """JPEG frames of one episode. ``frames[camera, t]`` decodes one image to (H, W, 3) uint8."""

    def __init__(self, path: Path):
        with np.load(path) as data:
            self._data = {camera: data[f"{camera}_data"] for camera in CAMERAS}
            self._offsets = {camera: data[f"{camera}_offsets"] for camera in CAMERAS}
        self.length = len(self._offsets["main"]) - 1

    def jpeg(self, camera: str, t: int) -> bytes:
        offsets = self._offsets[camera]
        return self._data[camera][offsets[t] : offsets[t + 1]].tobytes()

    def __getitem__(self, key: tuple[str, int]) -> np.ndarray:
        camera, t = key
        if not 0 <= t < self.length:
            raise IndexError(f"Timestep {t} outside episode of length {self.length}.")
        return np.asarray(Image.open(io.BytesIO(self.jpeg(camera, t))).convert("RGB"))


def _encode_jpegs(images: np.ndarray, max_side: int, quality: int) -> tuple[np.ndarray, np.ndarray]:
    chunks, offsets = [], [0]
    for image in images:
        pil = Image.fromarray(image)
        scale = max_side / max(pil.size)
        if scale < 1:
            pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.BILINEAR)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality)
        chunks.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
        offsets.append(offsets[-1] + len(chunks[-1]))
    return np.concatenate(chunks), np.asarray(offsets, dtype=np.int64)


class ActivationWriter:
    """Write episodes into an activation directory.

    ``collection`` describes how the activations are produced. If the directory
    already has a ``collection.json`` it must match, so that shards of one
    collection can share a directory but different collections cannot be mixed.
    """

    def __init__(self, root: str | Path, collection: dict, frame_max_side: int = 224, jpeg_quality: int = 90):
        self.root = Path(root)
        self.layers = collection["layers"]
        self.frame_max_side = frame_max_side
        self.jpeg_quality = jpeg_quality
        self.root.mkdir(parents=True, exist_ok=True)
        collection_path = self.root / COLLECTION_FILE
        if collection_path.exists():
            existing = json.loads(collection_path.read_text())
            if existing != collection:
                raise ValueError(
                    f"{collection_path} describes a different collection:\n  existing: {existing}\n"
                    f"  new:      {collection}\nUse a new output directory."
                )
        else:
            # One temporary file per process: shards started together all reach this branch.
            tmp = collection_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(collection, indent=2))
            tmp.replace(collection_path)
        for sub in ["episodes", "frames", *self.layers]:
            (self.root / sub).mkdir(exist_ok=True)

    def is_done(self, episode_id: int, need_activations: bool, need_frames: bool) -> bool:
        name = _episode_name(episode_id)
        if not (self.root / "episodes" / f"{name}.json").exists():
            return False
        if need_frames and not (self.root / "frames" / f"{name}.npz").exists():
            return False
        if need_activations:
            return all((self.root / layer / f"{name}.npy").exists() for layer in self.layers)
        return True

    def write_activations(self, episode_id: int, activations: dict[str, np.ndarray]):
        for layer in self.layers:
            np.save(self.root / layer / f"{_episode_name(episode_id)}.npy", activations[layer].astype(np.float32))

    def write_frames(self, episode_id: int, images: np.ndarray, wrist_images: np.ndarray):
        arrays = {}
        for camera, frames in zip(CAMERAS, (images, wrist_images)):
            arrays[f"{camera}_data"], arrays[f"{camera}_offsets"] = _encode_jpegs(
                frames, self.frame_max_side, self.jpeg_quality
            )
        np.savez(self.root / "frames" / f"{_episode_name(episode_id)}.npz", **arrays)

    def write_episode_meta(self, episode_id: int, task: str, length: int, source: dict):
        # Written last: its presence marks the episode as complete.
        meta = {"episode_id": episode_id, "task": task, "length": length, "source": source}
        (self.root / "episodes" / f"{_episode_name(episode_id)}.json").write_text(json.dumps(meta))
