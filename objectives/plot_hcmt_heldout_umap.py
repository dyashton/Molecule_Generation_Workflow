#!/usr/bin/env python3
"""Held-out latent maps for HCMT arms (not in the chemical tether).

Two feature spaces, same molecules:
  - MACCS keys → UMAP (Jaccard) — structural, outside Morgan/ChemBERTa/Uni-Mol
  - PhysChem (MW, LogP, TPSA, HBD, HBA, rotatable) → UMAP (Euclidean)

Writes:
  - <root>/hcmt_heldout_umap.png — bank + arm scatter with KDE contours overlaid
  - <root>/hcmt_heldout_umap_contours.png — small multiples (one arm per panel)

Example:
  PYTHONPATH=. uv run python objectives/plot_hcmt_heldout_umap.py \\
    --root outputs/unbounded_proxy_hacking_mw400/mollogp \\
    --arms no_tether dstar_100 dstar_130 dstar_200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import umap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from objectives.plot_hcmt_dstar_novelty import _late_unique_smiles
from objectives.plot_hcmt_dstar_tsne import (
    ARM_LABELS,
    COLORS,
    DEFAULT_BANK,
    _latest_run,
    _load_bank_smiles,
)

DEFAULT_ROOT = Path("outputs/unbounded_proxy_hacking_mw400/mollogp")
DEFAULT_ARMS = ("no_tether", "dstar_100", "dstar_130", "dstar_200")
EXTRA_LABELS = {"no_tether": "No tether"}
# Distinct from pretrained blue (#4C78A8)
EXTRA_COLORS = {"no_tether": "#2A9D8F"}
BANK_COLOR = "#B0B0B0"
MIN_KDE = 5


def _label(src: str) -> str:
    if src == "pretrained":
        return "Pretrained bank"
    return EXTRA_LABELS.get(src) or ARM_LABELS.get(src, src)


def _color(src: str) -> str:
    if src == "pretrained":
        return BANK_COLOR
    return EXTRA_COLORS.get(src) or COLORS.get(src, "#333333")


def _maccs_matrix(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    rows: list[np.ndarray] = []
    keep: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        keep.append(i)
    if not rows:
        raise ValueError("No valid MACCS fingerprints")
    return np.stack(rows, axis=0), keep


def _physchem_matrix(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    rows: list[np.ndarray] = []
    keep: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append(
            np.array(
                [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Lipinski.NumRotatableBonds(mol),
                ],
                dtype=np.float64,
            )
        )
        keep.append(i)
    if not rows:
        raise ValueError("No valid physchem rows")
    return np.stack(rows, axis=0), keep


def _umap_embed(X: np.ndarray, *, metric: str, seed: int) -> np.ndarray:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    n = X.shape[0]
    n_neighbors = int(min(30, max(5, n // 5)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric=metric,
        random_state=seed,
        n_jobs=1,
    )
    return reducer.fit_transform(X)


def _strip(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bank_scatter(ax, emb: np.ndarray, labels: list[str]) -> None:
    m = np.asarray([lab == "pretrained" for lab in labels])
    if not m.any():
        return
    ax.scatter(
        emb[m, 0],
        emb[m, 1],
        c=BANK_COLOR,
        s=14,
        alpha=0.28,
        marker="^",
        linewidths=0.0,
        label=_label("pretrained"),
        zorder=1,
    )


def _arm_contours(
    ax,
    emb: np.ndarray,
    labels: list[str],
    arms: list[str],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    draw_points: bool = True,
    levels: int = 4,
) -> None:
    """KDE contour outlines per arm; scatter fallback when n is tiny."""
    if xlim is None:
        xlim = (float(np.min(emb[:, 0])), float(np.max(emb[:, 0])))
    if ylim is None:
        ylim = (float(np.min(emb[:, 1])), float(np.max(emb[:, 1])))
    pad_x = 0.05 * (xlim[1] - xlim[0] + 1e-9)
    pad_y = 0.05 * (ylim[1] - ylim[0] + 1e-9)
    x0, x1 = xlim[0] - pad_x, xlim[1] + pad_x
    y0, y1 = ylim[0] - pad_y, ylim[1] + pad_y
    xx, yy = np.mgrid[x0:x1:120j, y0:y1:120j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for arm in arms:
        m = np.asarray([lab == arm for lab in labels])
        if int(m.sum()) == 0:
            continue
        pts = emb[m]
        color = _color(arm)
        name = _label(arm)
        if draw_points:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                s=22,
                alpha=0.45,
                linewidths=0.0,
                zorder=2,
            )
        if pts.shape[0] < MIN_KDE:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                s=48,
                alpha=0.85,
                linewidths=0.0,
                label=name,
                zorder=3,
            )
            continue
        try:
            kde = gaussian_kde(pts.T)
            zz = np.reshape(kde(grid), xx.shape)
            dens = kde(pts.T)
            qs = np.linspace(0.35, 0.92, max(int(levels), 2))
            lev = np.unique(np.quantile(dens, qs))
            if lev.size < 2:
                lev = np.linspace(float(zz.min()), float(zz.max()), 3)[1:]
            ax.contour(
                xx,
                yy,
                zz,
                levels=lev,
                colors=[color],
                linewidths=1.6,
                zorder=3,
            )
        except Exception:
            pass
        ax.plot([], [], color=color, lw=2.0, label=name)


def _save(fig, path: Path, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}", flush=True)
    figures = root.parent / "figures" / path.name
    if figures.parent.is_dir() and figures.resolve() != path.resolve():
        shutil.copy2(path, figures)
        print(f"Wrote {figures}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    p.add_argument("--n-pretrained", type=int, default=500)
    p.add_argument("--n-per-arm", type=int, default=500)
    p.add_argument("--n-snaps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out-contours", type=Path, default=None)
    args = p.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    bank = args.bank if args.bank.is_absolute() else ROOT / args.bank
    out = args.out
    if out is None:
        out = root / "hcmt_heldout_umap.png"
    elif not out.is_absolute():
        out = ROOT / out
    out_c = args.out_contours
    if out_c is None:
        out_c = root / "hcmt_heldout_umap_contours.png"
    elif not out_c.is_absolute():
        out_c = ROOT / out_c

    labels: list[str] = []
    smiles_all: list[str] = []

    pre = _load_bank_smiles(bank, args.n_pretrained, args.seed)
    labels.extend(["pretrained"] * len(pre))
    smiles_all.extend(pre)
    print(f"pretrained bank subsample: {len(pre)} from {bank}", flush=True)

    for arm in args.arms:
        run_dir = _latest_run(root / arm)
        arm_seed = args.seed + sum(ord(c) for c in arm)
        arm_smi = _late_unique_smiles(run_dir, args.n_snaps)
        if len(arm_smi) > args.n_per_arm:
            rng = np.random.default_rng(arm_seed)
            idx = rng.choice(len(arm_smi), size=args.n_per_arm, replace=False)
            arm_smi = [arm_smi[i] for i in sorted(idx)]
        labels.extend([arm] * len(arm_smi))
        smiles_all.extend(arm_smi)
        print(
            f"{arm}: {len(arm_smi)} unique from last healthy snaps "
            f"(n_snaps={args.n_snaps}, {run_dir.name})",
            flush=True,
        )

    Xm, keep_m = _maccs_matrix(smiles_all)
    labels_m = [labels[i] for i in keep_m]
    print(f"UMAP MACCS on {Xm.shape[0]} points...", flush=True)
    emb_m = _umap_embed(Xm, metric="jaccard", seed=args.seed)

    Xp_raw, keep_p = _physchem_matrix(smiles_all)
    labels_p = [labels[i] for i in keep_p]
    Xp = StandardScaler().fit_transform(Xp_raw)
    print(f"UMAP physchem on {Xp.shape[0]} points...", flush=True)
    emb_p = _umap_embed(Xp, metric="euclidean", seed=args.seed)

    arms = list(args.arms)
    xlim_m = (float(emb_m[:, 0].min()), float(emb_m[:, 0].max()))
    ylim_m = (float(emb_m[:, 1].min()), float(emb_m[:, 1].max()))
    xlim_p = (float(emb_p[:, 0].min()), float(emb_p[:, 0].max()))
    ylim_p = (float(emb_p[:, 1].min()), float(emb_p[:, 1].max()))

    # --- overlay: bank + all arm contours ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, emb, labs, xlim, ylim, title in (
        (axes[0], emb_m, labels_m, xlim_m, ylim_m, "MACCS → UMAP (held-out)"),
        (axes[1], emb_p, labels_p, xlim_p, ylim_p, "PhysChem → UMAP (held-out)"),
    ):
        _bank_scatter(ax, emb, labs)
        _arm_contours(ax, emb, labs, arms, xlim=xlim, ylim=ylim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(frameon=False, loc="best", fontsize=8)
        _strip(ax)
    fig.suptitle(
        "HCMT arms in held-out latents (KDE contours; last healthy snaps)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, out, root)

    # --- small multiples: one arm per column, MACCS / PhysChem rows ---
    n = len(arms)
    fig2, axes2 = plt.subplots(2, n, figsize=(3.2 * n, 6.4), sharex="row", sharey="row")
    if n == 1:
        axes2 = np.asarray(axes2).reshape(2, 1)
    for j, arm in enumerate(arms):
        for i, (emb, labs, xlim, ylim, row_name) in enumerate(
            (
                (emb_m, labels_m, xlim_m, ylim_m, "MACCS"),
                (emb_p, labels_p, xlim_p, ylim_p, "PhysChem"),
            )
        ):
            ax = axes2[i, j]
            _bank_scatter(ax, emb, labs)
            _arm_contours(
                ax,
                emb,
                labs,
                [arm],
                xlim=xlim,
                ylim=ylim,
                draw_points=True,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            _strip(ax)
            if i == 0:
                ax.set_title(_label(arm), fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{row_name}\nUMAP 2", fontsize=9)
            if i == 1:
                ax.set_xlabel("UMAP 1", fontsize=9)
            # Drop duplicate bank legend entries cluttering facets
            handles, labs_leg = ax.get_legend_handles_labels()
            keep = [
                (h, lab)
                for h, lab in zip(handles, labs_leg)
                if lab != _label("pretrained")
            ]
            if keep:
                ax.legend(
                    [h for h, _ in keep],
                    [lab for _, lab in keep],
                    frameon=False,
                    loc="best",
                    fontsize=7,
                )
    # one shared bank note
    fig2.suptitle(
        "Held-out UMAP contours by arm (gray △ = pretrained bank)",
        fontsize=12,
        y=1.01,
    )
    fig2.tight_layout()
    _save(fig2, out_c, root)


if __name__ == "__main__":
    main()
