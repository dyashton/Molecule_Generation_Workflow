#!/usr/bin/env python3
"""Literature-style novelty bars across HCMT D* arms vs the tether bank.

Late gated unique SMILES per arm:
  - SMILES novelty  = fraction not in bank (MOSES/GuacaMol novelty)
  - Scaffold novelty = fraction of Murcko scaffolds not in bank
  - Mean 1−SNN     = mean Morgan Tanimoto distance to nearest bank neighbor

Example:
  PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_novelty.py \\
    --root outputs/unbounded_proxy_hacking_mw400/mollogp \\
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
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from objectives.co2_rl_helpers import murcko_scaffold_smiles
from objectives.chemical_tether import tanimoto_nn_dist
from objectives.plot_hcmt_dstar_shells import _truthy
from objectives.plot_hcmt_dstar_tsne import (
    ARM_LABELS,
    COLORS,
    DEFAULT_BANK,
    _latest_run,
    _list_snapshots,
)

DEFAULT_ROOT = Path("outputs/unbounded_proxy_hacking_mw400/mollogp")
DEFAULT_ARMS = ("no_tether", "dstar_100", "dstar_130", "dstar_200")
EXTRA_LABELS = {"no_tether": "No tether"}
EXTRA_COLORS = {"no_tether": "#4C78A8"}


def _label(arm: str) -> str:
    return EXTRA_LABELS.get(arm) or ARM_LABELS.get(arm, arm)


def _color(arm: str) -> str:
    return EXTRA_COLORS.get(arm) or COLORS.get(arm, "#888888")


def _canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _scaffold(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaf = murcko_scaffold_smiles(mol)
    if scaf:
        return scaf
    # ponytail: acyclic → molecule itself as scaffold key (matches druglike_reward)
    return Chem.MolToSmiles(mol, canonical=True)


def _late_unique_smiles(
    run_dir: Path,
    n_snaps: int,
    *,
    min_gated: int = 16,
) -> list[str]:
    """Unique gated SMILES from last healthy snapshots (skip thin dying batches)."""
    snaps = _list_snapshots(run_dir)
    seen: set[str] = set()
    out: list[str] = []
    used = 0
    for _ep, path in reversed(snaps):
        if used >= max(int(n_snaps), 1):
            break
        df = pd.read_csv(path)
        if "smiles" not in df.columns:
            continue
        if "valid" in df.columns:
            df = df.loc[_truthy(df["valid"])]
        if "gate_ok" in df.columns:
            gated = df.loc[_truthy(df["gate_ok"])]
            if len(gated) > 0:
                df = gated
        if len(df) < int(min_gated):
            continue
        used += 1
        for smi in df["smiles"].astype(str):
            can = _canon(smi.strip())
            if can is None or can in seen:
                continue
            seen.add(can)
            out.append(can)
    if not out:
        raise ValueError(f"No healthy gated SMILES in snapshots of {run_dir}")
    return out


def _morgan_bits(smiles: list[str], n_bits: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    ok = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.uint8))
            ok.append(False)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        ok.append(True)
    return np.stack(rows, axis=0), np.asarray(ok, dtype=bool)


def _metrics(smiles: list[str], bank_set: set[str], bank_scaf: set[str], bank_morgan: np.ndarray) -> dict:
    n = len(smiles)
    novel = sum(1 for s in smiles if s not in bank_set)
    scafs = [_scaffold(s) for s in smiles]
    scafs_ok = [s for s in scafs if s]
    uniq_scaf = set(scafs_ok)
    novel_scaf = sum(1 for s in uniq_scaf if s not in bank_scaf)
    query, ok = _morgan_bits(smiles)
    nn = tanimoto_nn_dist(query[ok], bank_morgan, k=1) if ok.any() else np.array([])
    return {
        "n": n,
        "smiles_novelty": novel / n if n else float("nan"),
        "scaffold_novelty": novel_scaf / len(uniq_scaf) if uniq_scaf else float("nan"),
        "mean_1_snn": float(np.mean(nn)) if nn.size else float("nan"),
        "n_scaffolds": len(uniq_scaf),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    p.add_argument("--n-snaps", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    bank_path = args.bank if args.bank.is_absolute() else ROOT / args.bank
    out = args.out
    if out is None:
        out = root / "hcmt_dstar_novelty.png"
    elif not out.is_absolute():
        out = ROOT / out

    data = np.load(bank_path, allow_pickle=True)
    bank_smiles = [str(s) for s in data["smiles"].tolist()]
    bank_set = set()
    bank_scaf: set[str] = set()
    for smi in bank_smiles:
        can = _canon(smi)
        if can is None:
            continue
        bank_set.add(can)
        sc = _scaffold(can)
        if sc:
            bank_scaf.add(sc)
    bank_morgan = np.asarray(data["morgan"])
    print(f"bank n={len(bank_set)} scaffolds={len(bank_scaf)} from {bank_path}", flush=True)

    rows = []
    for arm in args.arms:
        arm_dir = root / arm
        if not arm_dir.is_dir():
            print(f"skip missing arm {arm}", flush=True)
            continue
        run_dir = _latest_run(arm_dir)
        smiles = _late_unique_smiles(run_dir, args.n_snaps)
        m = _metrics(smiles, bank_set, bank_scaf, bank_morgan)
        m["arm"] = arm
        rows.append(m)
        print(
            f"{arm}: n={m['n']} smiles_nov={m['smiles_novelty']:.3f} "
            f"scaf_nov={m['scaffold_novelty']:.3f} 1-SNN={m['mean_1_snn']:.3f} "
            f"({run_dir.name})",
            flush=True,
        )

    if not rows:
        raise SystemExit("No arms with data")

    arms = [r["arm"] for r in rows]
    labels = [_label(a) for a in arms]
    colors = [_color(a) for a in arms]
    x = np.arange(len(arms))

    panels = [
        ("smiles_novelty", "SMILES novelty vs bank", True),
        ("scaffold_novelty", "Scaffold novelty vs bank", True),
        ("mean_1_snn", "Mean 1 − SNN (Morgan)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=False)
    for ax, (key, title, as_pct) in zip(axes, panels):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        plot_vals = vals * 100.0 if as_pct else vals
        bars = ax.bar(x, plot_vals, color=colors, width=0.72, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title, loc="left", fontsize=11, fontweight="medium")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#ececec", lw=0.8)
        ax.set_axisbelow(True)
        if as_pct:
            ax.set_ylim(0, 105)
            ax.set_ylabel("%")
            for bar, v in zip(bars, plot_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            ymin = max(0.0, float(np.nanmin(plot_vals)) - 0.05)
            ax.set_ylim(ymin, float(np.nanmax(plot_vals)) + 0.08)
            for bar, v in zip(bars, plot_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    fig.suptitle("Novelty by HCMT arm (late gated snapshots)", fontsize=13, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    csv_out = out.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {csv_out}", flush=True)


if __name__ == "__main__":
    main()
