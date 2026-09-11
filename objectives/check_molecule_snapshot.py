#!/usr/bin/env python3
"""Smoke check for molecule snapshot CSV writer.

Run: PYTHONPATH=. uv run python objectives/check_molecule_snapshot.py
"""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

from reward import RewardBatch
from train import _write_molecule_snapshot


def main() -> None:
    batch = RewardBatch(
        policy=[1.0, 2.0],
        raw_total=[0.5, 1.5],
        raw_ba=[float("nan"), float("nan")],
        raw_div=[float("nan"), float("nan")],
        gate_ok=[True, False],
        raw_physchem=[0.4, 1.2],
        quality=[0.6, 1.4],
        mw_scale=[1.0, 0.8],
    )
    smiles = ["CCO", "not-a-smiles", "CCN"]
    valid_mask = [True, False, True]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ep5.csv"
        _write_molecule_snapshot(path, 5, smiles, batch, valid_mask)
        with path.open() as f:
            rows = list(csv.DictReader(f))

    assert len(rows) == 3, len(rows)
    assert list(rows[0].keys()) == [
        "episode",
        "smiles",
        "valid",
        "policy",
        "raw_total",
        "gate_ok",
        "raw_physchem",
        "quality",
        "mw_scale",
        "raw_ba",
        "raw_div",
    ]
    assert rows[0]["episode"] == "5"
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["valid"] == "True"
    assert float(rows[0]["policy"]) == 1.0
    assert rows[0]["gate_ok"] == "True"

    assert rows[1]["smiles"] == "not-a-smiles"
    assert rows[1]["valid"] == "False"
    assert math.isnan(float(rows[1]["policy"]))
    assert rows[1]["gate_ok"] == "False"

    assert rows[2]["smiles"] == "CCN"
    assert float(rows[2]["policy"]) == 2.0
    assert rows[2]["gate_ok"] == "False"
    print("ok: molecule snapshot CSV schema/rows")


if __name__ == "__main__":
    main()
