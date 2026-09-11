#!/usr/bin/env python3
"""Joint Morgan t-SNE: pretrained bank vs HCMT D* explore arms.

For each arm, locate the best checkpoint episode, take the nearest molecule
snapshot, then expand to neighboring snapshots (by |ep − best|) until enough
unique SMILES are collected (snapshots are batch-sized, ~64 rows).

Example:
  PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_tsne.py \\
    --root outputs/druglike_dstar_arms_nn
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = Path("outputs/druglike_dstar_arms_nn")
DEFAULT_BANK = Path("weights/chem_tether_bank.npz")
DEFAULT_ARMS = ("dstar_100", "dstar_130", "dstar_200")
ARM_LABELS = {
    "dstar_050": "D*=0.50",
    "dstar_080": "D*=0.80",
    "dstar_100": "D*=1.00",
    "dstar_130": "D*=1.30",
    "dstar_200": "D*=2.00",
}
COLORS = {
    "pretrained": "#4C78A8",
    "dstar_050": "#72B7B2",
    "dstar_080": "#54A24B",
    "dstar_100": "#F58518",
    "dstar_130": "#E45756",
    "dstar_200": "#B279A2",
}
_BEST_EP_RE = re.compile(r"best_.*_ep(\d+)\.pt$")


def _latest_run(arm_dir: Path) -> Path:
    candidates = list(arm_dir.rglob("rl_metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No rl_metrics.csv under {arm_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def _best_episode(run_dir: Path) -> int:
    """Episode of best_*.pt (via best.pt symlink or newest best_* file)."""
    ckpt_dir = run_dir / "checkpoints"
    best_link = ckpt_dir / "best.pt"
    target: Path | None = None
    if best_link.exists() or best_link.is_symlink():
        try:
            target = best_link.resolve()
        except OSError:
            target = None
    if target is None or not _BEST_EP_RE.search(target.name):
        bests = sorted(ckpt_dir.glob("best_*_ep*.pt"), key=lambda p: p.stat().st_mtime)
        if not bests:
            raise FileNotFoundError(f"No best_*_ep*.pt under {ckpt_dir}")
        target = bests[-1]
    m = _BEST_EP_RE.search(target.name)
    if not m:
        raise ValueError(f"Cannot parse best episode from {target.name}")
    return int(m.group(1))


def _list_snapshots(run_dir: Path) -> list[tuple[int, Path]]:
    snap_dir = run_dir / "molecule_snapshots"
    paths = list(snap_dir.glob("ep*.csv"))
    if not paths:
        raise FileNotFoundError(f"No molecule_snapshots under {run_dir}")
    return sorted(((int(p.stem[2:]), p) for p in paths), key=lambda t: t[0])


def _smiles_from_snapshot(path: Path, *, prefer_gated: bool) -> list[str]:
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"{path} missing smiles column")
    if "valid" in df.columns:
        m = df["valid"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        df = df.loc[m]
    if prefer_gated and "gate_ok" in df.columns:
        m = df["gate_ok"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        gated = df.loc[m]
        if len(gated) > 0:
            df = gated
    return (
        df["smiles"]
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s.str.len() > 0]
        .tolist()
    )


def _load_arm_near_best(
    run_dir: Path,
    *,
    n: int,
    seed: int,
    episode_override: int | None,
) -> tuple[list[str], int, list[int]]:
    """Unique SMILES from snapshots nearest to best (or --episode) until ``n``.

    Returns (smiles, center_episode, snapshot_eps_used).
    """
    snaps = _list_snapshots(run_dir)
    center = int(episode_override) if episode_override is not None else _best_episode(run_dir)
    # Absolute nearest first; ties prefer snapshot ≤ center (policy already trained to that ep).
    ordered = sorted(
        snaps,
        key=lambda t: (abs(t[0] - center), 0 if t[0] <= center else 1, -t[0]),
    )

    seen: set[str] = set()
    ordered_unique: list[str] = []
    used_eps: list[int] = []

    def _ingest(prefer_gated: bool) -> None:
        for ep, path in ordered:
            if len(ordered_unique) >= n:
                return
            before = len(ordered_unique)
            for smi in _smiles_from_snapshot(path, prefer_gated=prefer_gated):
                if smi in seen:
                    continue
                seen.add(smi)
                ordered_unique.append(smi)
                if len(ordered_unique) >= n:
                    break
            if len(ordered_unique) > before:
                used_eps.append(ep)

    # Prefer gated; if still short, reopen without gate filter.
    _ingest(prefer_gated=True)
    if len(ordered_unique) < n:
        _ingest(prefer_gated=False)

    if not ordered_unique:
        raise ValueError(f"No usable SMILES near ep{center} under {run_dir}")

    rng = np.random.default_rng(seed)
    if len(ordered_unique) > n:
        idx = rng.choice(len(ordered_unique), size=n, replace=False)
        ordered_unique = [ordered_unique[i] for i in sorted(idx)]
    return ordered_unique, center, used_eps


def _load_bank_smiles(bank: Path, n: int, seed: int) -> list[str]:
    data = np.load(bank, allow_pickle=True)
    smiles = [str(s) for s in data["smiles"].tolist()]
    rng = np.random.default_rng(seed)
    if len(smiles) > n:
        idx = rng.choice(len(smiles), size=n, replace=False)
        smiles = [smiles[i] for i in idx]
    return smiles


def _morgan_matrix(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    rows: list[np.ndarray] = []
    keep: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        keep.append(i)
    if not rows:
        raise ValueError("No valid Morgan fingerprints")
    return np.stack(rows, axis=0), keep


def _fit_tsne(Xp: np.ndarray, seed: int) -> np.ndarray:
    # Single-thread BLAS/OpenMP to avoid hangs after CUDA train jobs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, Xp.shape[0] // 4)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
        n_jobs=1,
    )
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            return tsne.fit_transform(Xp)
    except ImportError:
        return tsne.fit_transform(Xp)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    p.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Override best-ckpt episode (still expands nearby snapshots)",
    )
    p.add_argument("--n-pretrained", type=int, default=500)
    p.add_argument("--n-per-arm", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <root>/hcmt_dstar_tsne.png)",
    )
    args = p.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    bank = args.bank if args.bank.is_absolute() else ROOT / args.bank
    out = args.out
    if out is None:
        out = root / "hcmt_dstar_tsne.png"
    elif not out.is_absolute():
        out = ROOT / out

    labels: list[str] = []
    smiles_all: list[str] = []

    pre = _load_bank_smiles(bank, args.n_pretrained, args.seed)
    labels.extend(["pretrained"] * len(pre))
    smiles_all.extend(pre)
    print(f"pretrained bank subsample: {len(pre)} from {bank}", flush=True)

    for arm in args.arms:
        arm_dir = root / arm
        run_dir = _latest_run(arm_dir)
        arm_seed = args.seed + sum(ord(c) for c in arm)
        arm_smi, center, used = _load_arm_near_best(
            run_dir,
            n=args.n_per_arm,
            seed=arm_seed,
            episode_override=args.episode,
        )
        labels.extend([arm] * len(arm_smi))
        smiles_all.extend(arm_smi)
        nearest = used[0] if used else center
        print(
            f"{arm}: {len(arm_smi)} unique SMILES "
            f"(best/center ep={center}, nearest snap=ep{nearest}, "
            f"expanded {len(used)} snapshots: {used[:8]}{'...' if len(used) > 8 else ''})",
            flush=True,
        )
        if len(arm_smi) < args.n_per_arm:
            print(
                f"  warning: only {len(arm_smi)}/{args.n_per_arm} available near best",
                flush=True,
            )

    X, keep = _morgan_matrix(smiles_all)
    labels = [labels[i] for i in keep]
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    Xp = PCA(n_components=n_comp, random_state=args.seed).fit_transform(X)
    print(f"Fitting t-SNE on {Xp.shape[0]} points (PCA→{n_comp})...", flush=True)
    emb = _fit_tsne(Xp, args.seed)
    print("t-SNE done.", flush=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    order = ["pretrained", *args.arms]
    for src in order:
        m = np.asarray([lab == src for lab in labels])
        if not m.any():
            continue
        label = "Pretrained bank" if src == "pretrained" else ARM_LABELS.get(src, src)
        ax.scatter(
            emb[m, 0],
            emb[m, 1],
            c=COLORS.get(src, "#333333"),
            s=28 if src == "pretrained" else 36,
            alpha=0.35 if src == "pretrained" else 0.55,
            marker="^" if src == "pretrained" else "o",
            linewidths=0.0,
            label=label,
            zorder=1 if src == "pretrained" else 2,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        "HCMT novelty shells vs pretrained (best-ckpt snapshots; Morgan → PCA → t-SNE)"
    )
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
