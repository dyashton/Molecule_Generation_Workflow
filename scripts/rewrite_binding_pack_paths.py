#!/usr/bin/env python3
"""Point pack.joblib head paths at a local artifacts/binding_predictor dir."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("artifacts/binding_predictor"),
    )
    args = parser.parse_args()
    dest = args.dest.resolve()
    pack_path = dest / "pack.joblib"
    pack = joblib.load(pack_path)
    changed = False
    for key in ("ie_path", "be_path", "penalty_path", "strong_path", "begm_direct_path"):
        if key not in pack:
            continue
        name = Path(str(pack[key])).name
        local = dest / name
        if local.is_file():
            pack[key] = str(local)
            changed = True
    pack["reward_head"] = "ie"
    joblib.dump(pack, pack_path)
    print(f"[fetch] pack ready at {pack_path} (reward_head=ie, rewritten={changed})")


if __name__ == "__main__":
    main()
