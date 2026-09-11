"""
Inference: sample SMILES with a checkpoint, score with `reward.compute_rewards`.
Writes raw_smiles.csv and scored_smiles.csv under inference_outputs/<RUN_NAME>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from reward import compute_rewards
from transpharmer.generate import generate_yield
from transpharmer.SMILEValidifier import validate_smiles
from transpharmer.utils.io import load_config


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve(config_dir: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (config_dir / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score molecules")
    parser.add_argument("--config", default="generate.yaml")
    parser.add_argument("--ckpt", default=None, help="Override GENERATE.CKPT_PATH")
    parser.add_argument("--run-name", default=None, help="Override RUN_NAME")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (_script_dir() / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    config_dir = cfg_path.parent
    config = load_config(str(cfg_path))

    run_name = args.run_name or str(getattr(config, "RUN_NAME", "inference_run"))
    out_root = _resolve(config_dir, str(config.OUTPUTS.DIR))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = run_dir / "raw_smiles.csv"
    scored_csv = run_dir / "scored_smiles.csv"

    if args.ckpt:
        config.GENERATE.CKPT_PATH = str(_resolve(config_dir, args.ckpt))
    else:
        config.GENERATE.CKPT_PATH = str(_resolve(config_dir, config.GENERATE.CKPT_PATH))

    config.GENERATE.OUTPUT = str(raw_csv)

    rows: list[dict] = []
    seen: set[str] = set()

    stream = generate_yield(config)
    for gen_smiles in stream:
        for smi in validate_smiles(gen_smiles):
            if smi in seen:
                continue
            seen.add(smi)
            r = compute_rewards([smi])[0]
            rows.append({"SMILES": smi, "reward": r})

    if rows:
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["SMILES"], keep="last")
        df = df.sort_values(by="reward", ascending=False)
        df.to_csv(scored_csv, index=False)
        print(f"Scored {len(df)} unique SMILES -> {scored_csv}")
    else:
        pd.DataFrame(columns=["SMILES", "reward"]).to_csv(scored_csv, index=False)
        print("No valid SMILES produced.")

    if raw_csv.exists():
        print(f"Raw generation log: {raw_csv}")


if __name__ == "__main__":
    main()
