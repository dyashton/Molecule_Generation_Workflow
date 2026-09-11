"""Offline, matched-checkpoint evaluation for the chemical tether.

Every arm is scored against the same frozen reference bank and deterministic
held-out split.  The generated policy does not need to have used the tether.

Examples:
    uv run --extra tether python objectives/evaluate_chemical_tether.py \
        --root outputs/ablation_cpu_nopenalty --output outputs/tether_eval.csv

    uv run python objectives/evaluate_chemical_tether.py \
        --root outputs/ablation_cpu_nopenalty --morgan-only
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from objectives.chemical_tether import ChemicalTether, ChemBERTaEncoder, UniMolEncoder


DEFAULT_ROOT = Path("outputs/ablation_cpu_nopenalty")
DEFAULT_BANK = Path("weights/chem_tether_bank.npz")
DEFAULT_CHECKPOINTS = (100, 250, 500, 750, 1000)


def _canonical(smiles: object) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _latest_run(root: Path, arm: str) -> Path:
    candidates = list((root / arm).rglob("rl_metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No rl_metrics.csv found for {root / arm}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def _snapshot_for(run_dir: Path, target: int) -> tuple[Path, int] | None:
    paths = sorted(
        (run_dir / "molecule_snapshots").glob("ep*.csv"),
        key=lambda path: int(path.stem[2:]),
    )
    eligible = [path for path in paths if int(path.stem[2:]) <= target]
    if not eligible:
        return None
    path = eligible[-1]
    return path, int(path.stem[2:])


def _split_bank(
    bank_path: Path, heldout_fraction: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = np.load(bank_path, allow_pickle=True)
    smiles = [str(value) for value in data["smiles"].tolist()]
    keys = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    (_canonical(smi) or smi).encode("utf-8")
                ).digest()[:8],
                "big",
            )
            for smi in smiles
        ],
        dtype=np.uint64,
    )
    # Seed is included in the deterministic hash so the split can be changed
    # deliberately without depending on row order.
    threshold = int(heldout_fraction * (2**64 - 1))
    split_key = keys ^ np.uint64(seed)
    heldout = split_key <= threshold
    if not heldout.any() or heldout.all():
        raise ValueError("Held-out fraction produced an empty bank split")

    def take(mask: np.ndarray) -> dict[str, np.ndarray]:
        out = {
            "smiles": np.asarray(smiles, dtype=object)[mask],
            "morgan": np.asarray(data["morgan"])[mask],
        }
        for key in ("emb_2d", "emb_3d"):
            if key in data.files:
                out[key] = np.asarray(data[key])[mask]
        return out

    return take(~heldout), take(heldout)


def _make_tether(
    bank: dict[str, np.ndarray],
    *,
    morgan_only: bool,
    encoder_2d: ChemBERTaEncoder | None,
    encoder_3d: UniMolEncoder | None,
) -> ChemicalTether:
    return ChemicalTether(
        smiles=[str(value) for value in bank["smiles"].tolist()],
        morgan=bank["morgan"],
        emb_2d=None if morgan_only else bank.get("emb_2d"),
        emb_3d=None if morgan_only else bank.get("emb_3d"),
        lambda_2d=0.0 if morgan_only else 1.0,
        lambda_3d=0.0 if morgan_only else 1.0,
        encoder_2d=encoder_2d,
        encoder_3d=encoder_3d,
    )


def _distance_summary(
    tether: ChemicalTether,
    smiles: list[str],
    indices: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if len(indices) == 0:
        return {f"{prefix}_{key}": float("nan") for key in (
            "combined_median",
            "combined_p90",
            "morgan_median",
            "morgan_p90",
            "d2d_median",
            "d3d_median",
        )}
    selected = [smiles[int(i)] for i in indices]
    combined, channels = tether.distances(selected)

    def median(values: np.ndarray) -> float:
        return float(np.nanmedian(values))

    def p90(values: np.ndarray) -> float:
        return float(np.nanpercentile(values, 90))

    return {
        f"{prefix}_combined_median": median(combined.numpy()),
        f"{prefix}_combined_p90": p90(combined.numpy()),
        f"{prefix}_morgan_median": median(channels["morgan"]),
        f"{prefix}_morgan_p90": p90(channels["morgan"]),
        f"{prefix}_d2d_median": median(channels["2d"]),
        f"{prefix}_d3d_median": median(channels["3d"]),
    }


def _evaluate_snapshot(
    path: Path,
    *,
    run_label: str,
    arm: str,
    target_episode: int,
    actual_episode: int,
    reference: dict[str, np.ndarray],
    heldout: dict[str, np.ndarray],
    reference_tether: ChemicalTether,
    heldout_tether: ChemicalTether,
) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"smiles", "valid", "gate_ok", "quality"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    smiles = frame["smiles"].astype(str).tolist()
    canonical = [_canonical(smi) for smi in smiles]
    valid = _truthy(frame["valid"]).to_numpy()
    gated = valid & _truthy(frame["gate_ok"]).to_numpy()
    finite_quality = np.isfinite(pd.to_numeric(frame["quality"], errors="coerce"))
    parseable = np.asarray([value is not None for value in canonical])
    bank_smiles = set(reference["smiles"].tolist()) | set(heldout["smiles"].tolist())
    unique_gated = {canonical[i] for i in np.flatnonzero(gated & parseable)}

    rows = []
    for scope, indices in (
        ("all", np.flatnonzero(parseable)),
        ("gated", np.flatnonzero(gated & finite_quality & parseable)),
    ):
        row: dict[str, object] = {
            "run": run_label,
            "arm": arm,
            "target_episode": target_episode,
            "actual_episode": actual_episode,
            "scope": scope,
            "n_generated": len(frame),
            "n_parseable": int(parseable.sum()),
            "n_valid": int(valid.sum()),
            "n_gated": int(gated.sum()),
            "valid_fraction": float(valid.mean()) if len(valid) else float("nan"),
            "gate_fraction": float(gated.mean()) if len(gated) else float("nan"),
            "n_unique_gated": len(unique_gated),
            "unique_gated_fraction": (
                len(unique_gated) / int(gated.sum()) if gated.any() else float("nan")
            ),
            "bank_exact_overlap": len(unique_gated & bank_smiles),
        }
        row.update(
            _distance_summary(
                reference_tether, smiles, indices, "reference"
            )
        )
        row.update(
            _distance_summary(
                heldout_tether, smiles, indices, "heldout"
            )
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="Ablation root; repeat to compare standard and no-penalty runs.",
    )
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=Path("outputs/tether_evaluation.csv"))
    parser.add_argument(
        "--checkpoints",
        default=",".join(map(str, DEFAULT_CHECKPOINTS)),
        help="Comma-separated matched target episodes.",
    )
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--morgan-only", action="store_true")
    args = parser.parse_args()

    roots = args.root or [DEFAULT_ROOT]
    checkpoints = tuple(int(value) for value in args.checkpoints.split(",") if value)
    reference, heldout = _split_bank(
        args.bank, args.heldout_fraction, args.split_seed
    )
    encoder_2d = None if args.morgan_only else ChemBERTaEncoder(device="cpu")
    encoder_3d = None if args.morgan_only else UniMolEncoder(device="cpu")
    reference_tether = _make_tether(
        reference,
        morgan_only=args.morgan_only,
        encoder_2d=encoder_2d,
        encoder_3d=encoder_3d,
    )
    heldout_tether = _make_tether(
        heldout,
        morgan_only=args.morgan_only,
        encoder_2d=encoder_2d,
        encoder_3d=encoder_3d,
    )

    rows: list[dict[str, object]] = []
    for root in roots:
        root = root.resolve()
        arms = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and list(path.rglob("rl_metrics.csv"))
        )
        for arm in arms:
            run_dir = _latest_run(root, arm)
            for target in checkpoints:
                selected = _snapshot_for(run_dir, target)
                if selected is None:
                    continue
                path, actual = selected
                rows.extend(
                    _evaluate_snapshot(
                        path,
                        run_label=root.name,
                        arm=arm,
                        target_episode=target,
                        actual_episode=actual,
                        reference=reference,
                        heldout=heldout,
                        reference_tether=reference_tether,
                        heldout_tether=heldout_tether,
                    )
                )

    if not rows:
        raise SystemExit("No compatible molecule snapshots found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(
        f"Wrote {len(rows)} rows to {args.output} "
        f"(reference={len(reference['smiles'])}, heldout={len(heldout['smiles'])}, "
        f"morgan_only={args.morgan_only})"
    )


if __name__ == "__main__":
    main()
