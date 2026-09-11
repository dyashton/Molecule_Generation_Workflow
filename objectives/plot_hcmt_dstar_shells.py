#!/usr/bin/env python3
"""Calibrated d' violins for HCMT D* arms vs bank leave-one-out.

This is the shell the tether actually uses (NN-units, typical bank neighbor = 1).
Late snapshots, one violin per arm, plus the pretrained bank. Vertical lines at
each arm's D*. Facets: combined / Morgan / 2D / 3D.

Example:
  PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_shells.py \\
    --root outputs/druglike_dstar_arms_nn_c4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from objectives.chemical_tether import loo_channel_distances
from objectives.plot_hcmt_dstar_tsne import (
    ARM_LABELS,
    COLORS,
    DEFAULT_ARMS,
    DEFAULT_BANK,
    _best_episode,
    _latest_run,
    _list_snapshots,
)

DEFAULT_ROOT = Path("outputs/druglike_dstar_arms_nn_c4")
ARM_DSTAR = {
    "dstar_050": 0.50,
    "dstar_080": 0.80,
    "dstar_100": 1.00,
    "dstar_130": 1.30,
    "dstar_200": 2.00,
}
EXTRA_LABELS = {"no_tether": "No tether"}
EXTRA_COLORS = {"no_tether": "#4C78A8"}
DP_COLS = {
    "morgan": "tether_dp_morgan",
    "2d": "tether_dp_2d",
    "3d": "tether_dp_3d",
}
PANEL_TITLES = {
    "combined": "Combined d′",
    "morgan": "Morgan d′",
    "2d": "ChemBERTa 2D d′",
    "3d": "Uni-Mol 3D d′",
}


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _combined(frame: pd.DataFrame) -> np.ndarray:
    parts = [pd.to_numeric(frame[col], errors="coerce") for col in DP_COLS.values()]
    stacked = np.vstack([p.to_numpy(dtype=np.float64) for p in parts])
    return np.nanmean(stacked, axis=0)


def _late_arm_dprime(
    run_dir: Path,
    n_snaps: int,
    *,
    min_gated: int = 16,
) -> dict[str, np.ndarray]:
    """Last healthy gated d′: walk back from end, skip thin/empty batches."""
    snaps = _list_snapshots(run_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots under {run_dir}")
    rows: list[pd.DataFrame] = []
    used = 0
    last_ep = -1
    for ep, path in reversed(snaps):
        if used >= max(int(n_snaps), 1):
            break
        df = pd.read_csv(path)
        missing = [c for c in DP_COLS.values() if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path} missing {missing}. Re-run with a bank that logs tether_dp_*."
            )
        if "valid" in df.columns:
            df = df.loc[_truthy(df["valid"])]
        if "gate_ok" in df.columns:
            gated = df.loc[_truthy(df["gate_ok"])]
            if len(gated) > 0:
                df = gated
        if len(df) < int(min_gated):
            continue
        keep = ["smiles", *DP_COLS.values()]
        rows.append(df[keep].copy())
        used += 1
        last_ep = ep
    if not rows:
        raise ValueError(
            f"No healthy gated snaps (min_gated={min_gated}) under {run_dir}"
        )
    frame = pd.concat(rows, ignore_index=True)
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    frame = frame.loc[frame["smiles"].str.len() > 0]
    frame = frame.drop_duplicates(subset=["smiles"], keep="last")
    out = {key: pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
           for key, col in DP_COLS.items()}
    out["combined"] = _combined(frame)
    out["_ref_ep"] = np.asarray([float(last_ep)])
    return out


def _bank_loo_dprime(bank: Path, k: int) -> dict[str, np.ndarray]:
    data = np.load(bank, allow_pickle=True)
    emb_2d = data["emb_2d"] if "emb_2d" in data.files else None
    emb_3d = data["emb_3d"] if "emb_3d" in data.files else None
    raw = loo_channel_distances(data["morgan"], emb_2d, emb_3d, k=k)
    scales = {}
    for key, store in (("morgan", "scale_morgan"), ("2d", "scale_2d"), ("3d", "scale_3d")):
        if store in data.files:
            scales[key] = float(np.asarray(data[store]).reshape(-1)[0])
        else:
            d = raw[key]
            finite = d[np.isfinite(d)]
            scales[key] = float(np.median(finite)) if finite.size else 1.0
        if not np.isfinite(scales[key]) or scales[key] <= 1e-12:
            scales[key] = 1.0
    primed = {key: raw[key] / scales[key] for key in ("morgan", "2d", "3d")}
    stacked = np.vstack([primed["morgan"], primed["2d"], primed["3d"]])
    primed["combined"] = np.nanmean(stacked, axis=0)
    return primed


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _style_violin(ax, bodies, keys: list[str]) -> None:
    for body, key in zip(bodies, keys):
        color = EXTRA_COLORS.get(key) or COLORS.get(key, "#444444")
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
        body.set_linewidth(0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--n-snaps", type=int, default=10, help="Trailing snapshots per arm.")
    parser.add_argument("--k", type=int, default=1, help="Bank leave-one-out k.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    bank = args.bank if args.bank.is_absolute() else ROOT / args.bank
    out = args.out
    if out is None:
        out = root / "hcmt_dstar_shells.png"
    elif not out.is_absolute():
        out = ROOT / out

    series_keys = ["pretrained", *args.arms]
    # Put no_tether at the top of horizontal violins (highest y).
    if "no_tether" in series_keys:
        series_keys = [k for k in series_keys if k != "no_tether"] + ["no_tether"]
    series_labels = {
        "pretrained": "Pretrained bank",
        **{
            arm: EXTRA_LABELS.get(arm) or ARM_LABELS.get(arm, arm)
            for arm in args.arms
        },
    }
    data: dict[str, dict[str, np.ndarray]] = {
        "pretrained": _bank_loo_dprime(bank, args.k),
    }
    n_bank = int(np.isfinite(data["pretrained"]["combined"]).sum())
    print(f"bank LOO n={n_bank} from {bank}", flush=True)

    for arm in args.arms:
        run_dir = _latest_run(root / arm)
        data[arm] = _late_arm_dprime(run_dir, args.n_snaps)
        ref_ep = int(data[arm].pop("_ref_ep")[0])
        n = int(np.isfinite(data[arm]["combined"]).sum())
        print(
            f"{arm}: n={n} unique gated from last healthy snaps "
            f"(newest healthy ep{ref_ep}, n_snaps={args.n_snaps}) ({run_dir.name})",
            flush=True,
        )

    dstars = [ARM_DSTAR[arm] for arm in args.arms if arm in ARM_DSTAR]
    panels = ("combined", "morgan", "2d", "3d")
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 10.5), sharex=True)
    ypos = np.arange(len(series_keys))

    for ax, panel in zip(axes, panels):
        samples = [_finite(data[key][panel]) for key in series_keys]
        vp = ax.violinplot(
            samples,
            positions=ypos,
            vert=False,
            widths=0.72,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        _style_violin(ax, vp["bodies"], series_keys)
        vp["cmedians"].set_color("#1a1a1a")
        vp["cmedians"].set_linewidth(1.1)
        for arm in args.arms:
            if arm not in ARM_DSTAR:
                continue
            ax.axvline(
                ARM_DSTAR[arm],
                color=COLORS.get(arm, "#888888"),
                ls="--",
                lw=1.0,
                alpha=0.85,
                zorder=0,
            )
        ax.set_yticks(ypos)
        ax.set_yticklabels([series_labels[k] for k in series_keys])
        ax.set_ylabel("")
        ax.set_title(PANEL_TITLES[panel], loc="left", fontsize=11, fontweight="medium")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#ececec", lw=0.8)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("d′  (1 = typical bank nearest neighbor)")
    if dstars:
        axes[0].set_xlim(left=0.0, right=max(2.6, max(dstars) + 0.4))
    fig.suptitle(
        "HCMT novelty shells on the tether coordinate (last healthy snapshots)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
