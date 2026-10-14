"""Derive an env-canonical prompt bundle without changing feature IDs or old artifacts."""

import argparse
import copy
import json
from pathlib import Path

from experiments.rq2.common import manifest, output_path, sha256, write_json


def canonical_bundle(bundle, input_sha256):
    result = copy.deepcopy(bundle)
    result["manifest"].update(
        format="rq2-pooled-features-v2", prompt_style="env-lowercase",
        source_features_sha256=input_sha256,
    )
    for pair in result["pairs"].values():
        for key in ("old_prompt", "new_prompt"):
            pair[key + "_raw_bddl"] = pair[key]
            # bddl.parsing.scan_tokens lowercases the complete source, then the
            # simulator joins language tokens with a single space.
            pair[key] = " ".join(pair[key].lower().split())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = output_path(args.out)
    source = json.loads(args.features.read_text())
    if source["manifest"].get("prompt_style") == "env-lowercase":
        raise ValueError("Already canonical; use the existing bundle")
    result = canonical_bundle(source, sha256(args.features))
    result["manifest"]["normalizer_sha256"] = sha256(Path(__file__))
    manifest(out / "manifest.json", result["manifest"])
    destination = out / "features.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError("Existing derived features differ; choose a new output directory")
    else:
        write_json(destination, result)
    print(f"Canonicalized prompts only; feature IDs/means unchanged: {destination}")


if __name__ == "__main__":
    main()
