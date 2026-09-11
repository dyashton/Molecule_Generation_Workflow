#!/usr/bin/env python3
"""Druglike objective / physchem comparison across HCMT D* arms.

Late gated snapshots: violins of the optimization terms (objective, logit QED,
−SCScore, MolLogP, −log scaffold freq) plus raw MW / MolLogP / combined d′.

Example:
  PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_physchem.py \\
    --root outputs/druglike_dstar_arms_nn_c4 \\
    --arms no_tether dstar_100 dstar_130 dstar_200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from objectives.plot_hcmt_dstar_shells import DP_COLS, _truthy
from objectives.plot_hcmt_dstar_tsne import (
    ARM_LABELS,
    COLORS,
    _best_episode,
    _latest_run,
    _list_snapshots,
)

DEFAULT_ROOT = Path("outputs/druglike_dstar_arms_nn_c4")
DEFAULT_ARMS = ("no_tether", "dstar_100", "dstar_130", "dstar_200")
EXTRA_LABELS = {"no_tether": "No tether"}
EXTRA_COLORS = {"no_tether": "#4C78A8"}

PANELS = (
    ("quality", "Objective raw_total", None),
    ("raw_ba", "logit(QED) (0.35)", None),
    ("raw_physchem", "−SCScore (0.25)", None),
    ("mw_scale", "MolLogP (0.15)", None),
    ("raw_div", "−log scaffold freq (0.25)", None),
    ("mw", "Molecular weight", None),
    ("logp", "MolLogP", (0.0, 5.0)),
    ("d_prime", "Combined d′", (1.0, 1.3, 2.0)),
)


def _label(key: str) -> str:
    return EXTRA_LABELS.get(key) or ARM_LABELS.get(key, key)


def _color(key: str) -> str:
    return EXTRA_COLORS.get(key) or COLORS.get(key, "#444444")


def _enrich_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mw: list[float] = []
    logp: list[float] = []
    for smi in frame["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mw.append(float("nan"))
            logp.append(float("nan"))
            continue
        mw.append(float(Descriptors.MolWt(mol)))
        logp.append(float(Descriptors.MolLogP(mol)))
    frame = frame.copy()
    frame["mw"] = mw
    frame["logp"] = logp
    if set(DP_COLS.values()).issubset(frame.columns):
        stacked = np.vstack(
            [
                pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
                for col in DP_COLS.values()
            ]
        )
        frame["d_prime"] = np.nanmean(stacked, axis=0)
    else:
        frame["d_prime"] = np.nan
    return frame


def _arm_frame_near_best(run_dir: Path, n_snaps: int) -> tuple[pd.DataFrame, int]:
    """Gated snapshot rows nearest to best checkpoint episode."""
    snaps = _list_snapshots(run_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots under {run_dir}")
    center = _best_episode(run_dir)
    ordered = sorted(
        snaps,
        key=lambda t: (abs(t[0] - center), 0 if t[0] <= center else 1, -t[0]),
    )
    rows: list[pd.DataFrame] = []
    used = 0
    for _ep, path in ordered:
        if used >= max(int(n_snaps), 1):
            break
        df = pd.read_csv(path)
        if "valid" in df.columns:
            df = df.loc[_truthy(df["valid"])]
        if "gate_ok" in df.columns:
            gated = df.loc[_truthy(df["gate_ok"])]
            if len(gated) > 0:
                df = gated
        if df.empty:
            continue
        rows.append(df)
        used += 1
    if not rows:
        raise ValueError(f"No gated rows near best ep{center} under {run_dir}")
    frame = pd.concat(rows, ignore_index=True)
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    frame = frame.loc[frame["smiles"].str.len() > 0]
    frame = frame.drop_duplicates(subset=["smiles"], keep="last")
    return _enrich_frame(frame), center


def _late_arm_frame(
    run_dir: Path,
    n_snaps: int,
    *,
    min_gated: int = 16,
) -> pd.DataFrame:
    """Last healthy gated snapshots: walk back from end, skip thin/empty batches."""
    snaps = _list_snapshots(run_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots under {run_dir}")
    rows: list[pd.DataFrame] = []
    used = 0
    for _ep, path in reversed(snaps):
        if used >= max(int(n_snaps), 1):
            break
        df = pd.read_csv(path)
        if "valid" in df.columns:
            df = df.loc[_truthy(df["valid"])]
        if "gate_ok" in df.columns:
            gated = df.loc[_truthy(df["gate_ok"])]
            if len(gated) > 0:
                df = gated
        if len(df) < int(min_gated):
            continue
        rows.append(df)
        used += 1
    if not rows:
        raise ValueError(
            f"No healthy gated snaps (min_gated={min_gated}) under {run_dir}"
        )
    frame = pd.concat(rows, ignore_index=True)
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    frame = frame.loc[frame["smiles"].str.len() > 0]
    frame = frame.drop_duplicates(subset=["smiles"], keep="last")
    return _enrich_frame(frame)


def _finite(values) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=np.float64)
    return arr[np.isfinite(arr)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--n-snaps", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    out = args.out
    if out is None:
        out = root / "hcmt_dstar_physchem.png"
    elif not out.is_absolute():
        out = ROOT / out
    csv_out = args.csv
    if csv_out is None:
        csv_out = root / "hcmt_dstar_physchem.csv"
    elif not csv_out.is_absolute():
        csv_out = ROOT / csv_out

    pools: dict[str, pd.DataFrame] = {}
    for arm in args.arms:
        run_dir = _latest_run(root / arm)
        frame = _late_arm_frame(run_dir, args.n_snaps)
        pools[arm] = frame
        print(
            f"{arm}: n={len(frame)} unique gated "
            f"(last {args.n_snaps} snaps, {run_dir.name})",
            flush=True,
        )

    summary_rows = []
    for arm, frame in pools.items():
        row = {"arm": arm, "n": len(frame)}
        for col, _title, _refs in PANELS:
            x = _finite(frame[col])
            row[f"{col}_median"] = float(np.median(x)) if x.size else float("nan")
            row[f"{col}_mean"] = float(np.mean(x)) if x.size else float("nan")
        summary_rows.append(row)
        print(
            f"  objective={row['quality_median']:.3f}  logitQED={row['raw_ba_median']:.3f}  "
            f"−SC={row['raw_physchem_median']:.3f}  d′={row['d_prime_median']:.3f}  "
            f"MW={row['mw_median']:.1f}",
            flush=True,
        )
    pd.DataFrame(summary_rows).to_csv(csv_out, index=False)

    n_panels = len(PANELS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.5, 12.5), sharex=False)
    ypos = np.arange(len(args.arms))

    for ax, (col, title, refs) in zip(axes, PANELS):
        samples = [_finite(pools[arm][col]) for arm in args.arms]
        vp = ax.violinplot(
            samples,
            positions=ypos,
            vert=False,
            widths=0.72,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for body, arm in zip(vp["bodies"], args.arms):
            c = _color(arm)
            body.set_facecolor(c)
            body.set_edgecolor(c)
            body.set_alpha(0.55)
            body.set_linewidth(0.8)
        vp["cmedians"].set_color("#1a1a1a")
        vp["cmedians"].set_linewidth(1.1)
        if refs is not None:
            for x in refs:
                ax.axvline(float(x), color="#bbbbbb", ls="--", lw=0.9, zorder=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([_label(a) for a in args.arms])
        ax.set_title(title, loc="left", fontsize=11, fontweight="medium")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#ececec", lw=0.8)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("value")
    fig.suptitle(
        "Druglike optimization targets across HCMT arms",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {csv_out}", flush=True)


if __name__ == "__main__":
    main()
