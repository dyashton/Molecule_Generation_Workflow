#!/usr/bin/env python3
"""Generate molecules and export AL/BO handoff CSV (proposals.csv)."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_VENDOR = ROOT / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from reward import compute_rewards  # noqa: E402
from transpharmer.generate import generate_yield  # noqa: E402
from transpharmer.SMILEValidifier import validate_smiles  # noqa: E402
from transpharmer.utils.io import load_config  # noqa: E402

PROPOSAL_FIELDS = (
    "SMILES",
    "reward",
    "reward_head",
    "ood_gated",
    "strong_prob",
    "binding",
    "uncertainty",
    "ie",
    "be",
    "begm",
)


def _resolve(config_dir: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (config_dir / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="generate_co2_binding.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n", type=int, default=0, help="Stop after N unique SMILES (0=all)")
    parser.add_argument(
        "--objective",
        default="co2_binding_surrogate",
        help="Sets MGW_OBJECTIVE if not already set",
    )
    args = parser.parse_args()

    os.environ.setdefault("MGW_OBJECTIVE", args.objective)
    os.environ.setdefault("MGW_BINDING_HEAD", "ie")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")

    config_dir = cfg_path.parent
    config = load_config(str(cfg_path))
    if args.ckpt:
        config.GENERATE.CKPT_PATH = str(_resolve(config_dir, args.ckpt))
    else:
        config.GENERATE.CKPT_PATH = str(_resolve(config_dir, config.GENERATE.CKPT_PATH))

    out = args.out
    if out is None:
        out = ROOT / "artifacts" / "discovery" / "proposals.csv"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_csv = out.parent / "raw_smiles.csv"
    config.GENERATE.OUTPUT = str(raw_csv)

    # Detailed scorer when using binding surrogate.
    detail = None
    reward_head = os.environ.get("MGW_BINDING_HEAD", "ie")
    if os.environ.get("MGW_OBJECTIVE", "").strip().lower() == "co2_binding_surrogate":
        from objectives.co2_binding_surrogate import get_binding_surrogate_scorer

        detail = get_binding_surrogate_scorer()
        reward_head = detail.reward_head

    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for gen_smiles in generate_yield(config):
        for smi in validate_smiles(gen_smiles):
            if smi in seen:
                continue
            seen.add(smi)
            reward = float(compute_rewards([smi])[0])
            pred = detail.predict_one(smi) if detail is not None else None
            ood = False
            if pred is not None:
                ood = bool(pred["ood"]) or not (
                    pred["uncertainty"] <= detail.max_uncertainty
                )
            elif reward != reward:  # NaN
                ood = True
            rows.append(
                {
                    "SMILES": smi,
                    "reward": "" if reward != reward else f"{reward:.8f}",
                    "reward_head": reward_head,
                    "ood_gated": int(ood),
                    "strong_prob": (
                        f"{pred['strong_prob']:.6f}" if pred is not None else ""
                    ),
                    "binding": f"{pred['binding']:.6f}" if pred is not None else "",
                    "uncertainty": (
                        f"{pred['uncertainty']:.6f}" if pred is not None else ""
                    ),
                    "ie": f"{pred['ie']:.6f}" if pred is not None else "",
                    "be": f"{pred['be']:.6f}" if pred is not None else "",
                    "begm": f"{pred['begm']:.6f}" if pred is not None else "",
                }
            )
            if args.n > 0 and len(rows) >= args.n:
                break
        if args.n > 0 and len(rows) >= args.n:
            break

    # Canonical dedupe already via seen; sort by reward descending (NaNs last).
    def sort_key(row: dict[str, object]) -> tuple[int, float]:
        raw = row["reward"]
        if raw == "" or raw is None:
            return (1, 0.0)
        return (0, -float(raw))

    rows.sort(key=sort_key)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PROPOSAL_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] {len(rows)} proposals -> {out}")


if __name__ == "__main__":
    main()
