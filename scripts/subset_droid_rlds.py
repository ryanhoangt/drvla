#!/usr/bin/env python3
"""Write a DROID v1.0.1 RLDS dataset that holds only the episodes of an episode list.

Streams every shard of the public release over HTTPS (nothing is downloaded to
disk), keeps the records whose ``<recording_folderpath>--<file_path>`` key is
listed, and writes them unchanged, in release order, into new shards with a
matching ``dataset_info.json``. The result is read by ``DroidEpisodes`` like the
full release, so pass it as ``--droid-rlds-dir`` to collect_activations.py.
For the paper's 2,000 episodes it is about 40 GB instead of 1.9 TB.

Finished source shards are recorded in <out>/.parts, so the script can be
interrupted and resumed. Once every source shard is done, the parts are renamed
to final shards and the metadata is written.

Example:
    python scripts/subset_droid_rlds.py --episodes data/droid_2k_episodes.json \
        --out /data/droid_2k/droid/1.0.1 --workers 16
"""

import argparse
import json
import logging
import shutil
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_URL = "https://storage.googleapis.com/gresearch/robotics/droid/1.0.1"
KEY_FEATURES = ("episode_metadata/recording_folderpath", "episode_metadata/file_path")
MAX_ATTEMPTS = 5

logger = logging.getLogger("subset_droid_rlds")


def fetch(url: str):
    return urllib.request.urlopen(url, timeout=120)


def read_exact(stream, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise EOFError(f"Stream ended after {len(data)} of {n} bytes.")
    return data


def iter_records(stream):
    """Yield the records of a TFRecord byte stream (length, length CRC, data, data CRC)."""
    while True:
        header = stream.read(12)
        if not header:
            return
        if len(header) != 12:
            raise EOFError("Stream ended inside a record header.")
        (length,) = struct.unpack("<Q", header[:8])
        record = read_exact(stream, length)
        read_exact(stream, 4)
        yield record


def episode_key(record: bytes) -> str:
    import tensorflow as tf

    spec = {name: tf.io.FixedLenFeature([], tf.string) for name in KEY_FEATURES}
    parsed = tf.io.parse_single_example(record, spec)
    folder, file_path = (parsed[name].numpy().decode() for name in KEY_FEATURES)
    return f"{folder}--{file_path}"


def filter_shard(source: str, filename: str, expected: int, wanted: set[str], parts: Path) -> dict:
    """Keep the wanted records of one source shard; returns (and saves) a summary of the part."""
    import tensorflow as tf

    stem = filename.rsplit("-", 3)[1]  # the source shard index of "...tfrecord-00012-of-02048"
    part_path, done_path = parts / f"{stem}.tfrecord", parts / f"{stem}.json"
    tmp_path = parts / f"{stem}.tfrecord.tmp"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            keys, num_records = [], 0
            with fetch(f"{source}/{filename}") as stream, tf.io.TFRecordWriter(str(tmp_path)) as writer:
                for record in iter_records(stream):
                    num_records += 1
                    key = episode_key(record)
                    if key in wanted:
                        writer.write(record)
                        keys.append(key)
            if num_records != expected:
                raise EOFError(f"{filename}: read {num_records} records, dataset_info.json lists {expected}.")
            break
        except Exception as error:  # network errors and truncated streams: retry the whole shard
            tmp_path.unlink(missing_ok=True)
            if attempt == MAX_ATTEMPTS:
                raise
            logger.warning("%s attempt %d failed (%s), retrying", filename, attempt, error)
            time.sleep(10 * attempt)
    if keys:
        tmp_path.rename(part_path)
    else:
        tmp_path.unlink()
    summary = {"source": filename, "keys": keys, "num_bytes": part_path.stat().st_size if keys else 0}
    done_path.write_text(json.dumps(summary))
    return summary


def finalize(out: Path, parts: Path, info: dict, features: str, summaries: list[dict]):
    """Rename the non-empty parts to tfds shards and write dataset_info.json / features.json."""
    split = info["splits"][0]
    template = split["filepathTemplate"]
    kept = [s for s in summaries if s["keys"]]
    for index, summary in enumerate(kept):
        stem = summary["source"].rsplit("-", 3)[1]
        shard = template.format(
            DATASET=info["name"], SPLIT=split["name"], FILEFORMAT=info["fileFormat"],
            SHARD_X_OF_Y=f"{index:05d}-of-{len(kept):05d}",
        )
        (parts / f"{stem}.tfrecord").rename(out / shard)
    split["shardLengths"] = [str(len(s["keys"])) for s in kept]
    split["numBytes"] = str(sum(s["num_bytes"] for s in kept))
    (out / "features.json").write_text(features)
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default=str(REPO_ROOT / "data" / "droid_2k_episodes.json"))
    parser.add_argument("--out", required=True, type=Path, help="Output directory, conventionally <root>/droid/1.0.1.")
    parser.add_argument("--source", default=SOURCE_URL, help="HTTP(S) URL of the droid/1.0.1 release.")
    parser.add_argument("--workers", type=int, default=16, help="Source shards streamed in parallel.")
    parser.add_argument("--shards", type=int, nargs="+", help="Only these source shard indices (for tests).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # parsing needs no GPU; do not take memory from a running policy

    wanted = {entry["key"] for entry in json.loads(Path(args.episodes).read_text())["episodes"]}
    with fetch(f"{args.source}/dataset_info.json") as response:
        info = json.load(response)
    with fetch(f"{args.source}/features.json") as response:
        features = response.read().decode()
    split = info["splits"][0]
    lengths = [int(n) for n in split["shardLengths"]]
    filenames = [
        split["filepathTemplate"].format(
            DATASET=info["name"], SPLIT=split["name"], FILEFORMAT=info["fileFormat"],
            SHARD_X_OF_Y=f"{index:05d}-of-{len(lengths):05d}",
        )
        for index in range(len(lengths))
    ]
    indices = args.shards if args.shards is not None else list(range(len(lengths)))

    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "dataset_info.json").exists():
        raise FileExistsError(f"{args.out} already holds a finished dataset.")
    parts = args.out / ".parts"
    parts.mkdir(exist_ok=True)
    summaries = {}
    for index in indices:
        done_path = parts / f"{index:05d}.json"
        if done_path.exists():
            summaries[index] = json.loads(done_path.read_text())
    todo = [index for index in indices if index not in summaries]
    logger.info("%d listed episodes; %d source shards, %d left to stream", len(wanted), len(indices), len(todo))

    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(filter_shard, args.source, filenames[i], lengths[i], wanted, parts): i for i in todo}
        for done, future in enumerate(as_completed(futures), 1):
            summaries[futures[future]] = future.result()
            found = sum(len(s["keys"]) for s in summaries.values())
            elapsed = time.time() - start
            logger.info(
                "%d/%d shards, %d/%d episodes found, %.1f min elapsed, ~%.1f min left",
                done, len(todo), found, len(wanted), elapsed / 60, elapsed / done * (len(todo) - done) / 60,
            )

    found = [key for index in indices for key in summaries[index]["keys"]]
    if len(found) != len(set(found)):
        raise ValueError("A listed key appears more than once in the release.")
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"{len(missing)} listed episodes were not found, e.g. {sorted(missing)[:3]}.")
    finalize(args.out, parts, info, features, [summaries[index] for index in sorted(indices)])
    logger.info("Wrote %d episodes to %s", len(found), args.out)


if __name__ == "__main__":
    main()
