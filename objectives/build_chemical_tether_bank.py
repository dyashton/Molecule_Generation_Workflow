#!/usr/bin/env python3
"""Build GuacaMol ∪ AxelRolov chemical-tether bank (Morgan + optional 2D/3D).

Run (after `uv sync --extra tether`):

    PYTHONPATH=. uv run python objectives/build_chemical_tether_bank.py

Morgan-only (no pretrained weights):

    PYTHONPATH=. uv run python objectives/build_chemical_tether_bank.py --morgan-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR = ROOT / "weights" / "prior_fp_bank.smi"
DEFAULT_AXEL = ROOT / "publication" / "co2" / "data" / "axelrolov_scored_nconf2.csv"
DEFAULT_OUT = ROOT / "weights" / "chem_tether_bank.npz"


def _load_axel(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"AxelRolov CSV not found: {path}")
    df = pd.read_csv(path)
    col = "SMILES" if "SMILES" in df.columns else "smiles"
    if col not in df.columns:
        raise ValueError(f"No SMILES column in {path}: {list(df.columns)}")
    return [str(s) for s in df[col].tolist() if isinstance(s, str) and s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chemical tether manifold bank")
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--axel", type=Path, default=DEFAULT_AXEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--morgan-only", action="store_true")
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-2d",
        default=None,
        help="HuggingFace ChemBERTa id (default: DeepChem/ChemBERTa-77M-MLM)",
    )
    args = parser.parse_args()

    from objectives.chemical_tether import (
        ChemBERTaEncoder,
        UniMolEncoder,
        morgan_bits,
        save_bank,
        union_manifold_smiles,
        DEFAULT_2D_MODEL,
        MORGAN_N_BITS,
    )
    from objectives.prior_fp_tether import load_smiles_cache, save_smiles_cache

    prior = load_smiles_cache(args.prior)
    axel = _load_axel(args.axel)
    smiles = union_manifold_smiles(prior, axel)
    if not smiles:
        print("No parseable SMILES in prior ∪ AxelRolov.", file=sys.stderr)
        sys.exit(1)
    print(
        f"Manifold: prior={len(prior)} axel={len(axel)} union={len(smiles)}",
        flush=True,
    )

    morgan = np.zeros((len(smiles), MORGAN_N_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        bits = morgan_bits(smi)
        if bits is None:
            continue
        morgan[i] = bits
    print(f"Morgan: {int(morgan.any(axis=1).sum())}/{len(smiles)} fps", flush=True)

    skip_2d = bool(args.morgan_only or args.skip_2d)
    skip_3d = bool(args.morgan_only or args.skip_3d)
    emb_2d = None
    emb_3d = None
    model_2d = args.model_2d or DEFAULT_2D_MODEL

    if not skip_2d:
        print(f"Encoding 2D ({model_2d})…", flush=True)
        enc = ChemBERTaEncoder(model_2d, device=args.device)
        emb_2d, ok2 = enc.encode(smiles)
        print(f"2D: {int(ok2.sum())}/{len(smiles)} dim={emb_2d.shape[1]}", flush=True)

    if not skip_3d:
        print("Encoding 3D (Uni-Mol)…", flush=True)
        enc3 = UniMolEncoder(device=args.device)
        emb_3d, ok3 = enc3.encode(smiles)
        print(f"3D: {int(ok3.sum())}/{len(smiles)} dim={emb_3d.shape[1]}", flush=True)

    scales = save_bank(args.out, smiles, morgan, emb_2d=emb_2d, emb_3d=emb_3d)
    sidecar = args.out.with_suffix(".smi")
    save_smiles_cache(sidecar, smiles)
    print(
        f"Wrote {args.out} and {sidecar}  "
        f"s_morgan={scales['morgan']:.4f} s_2d={scales['2d']:.4f} "
        f"s_3d={scales['3d']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
