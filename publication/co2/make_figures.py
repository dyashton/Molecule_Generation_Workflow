#!/usr/bin/env python3
"""Build CO2 publication figures under publication/co2/figures/.

Generates matched pretrained vs RL libraries (cached), scores CO2 components,
and writes the efficacy figure panel set.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from objectives.co2_rl_helpers import (  # noqa: E402
    amine_justified_oh_excess,
    consec_rot_scale,
    count_amine_n,
    count_oh,
    high_end_excess,
    max_consecutive_rotatable_bonds,
    mw_excess,
    mw_scale,
    oh_scale,
    sa_scale,
)
from objectives.co2_reward import (  # noqa: E402
    CONSEC_ROT_FULL_HI,
    CONSEC_ROT_ZERO_HI,
    MW_FLAT_HI,
    MW_FLAT_LO,
    MW_SCALE_HI,
    MW_SCALE_LO,
    OH_AMINE_FREE,
    OH_SCALE_FULL_HI,
    OH_SCALE_ZERO_HI,
    RANGES,
    SA_SCALE_FULL_HI,
    SA_SCALE_ZERO_HI,
    W_APOL,
    W_BASICITY,
    W_CONSEC_ROT_PENALTY,
    W_DIVERSE,
    W_HALL_KIER_ALPHA,
    W_LOGP,
    W_MW_PENALTY,
    W_OH_PENALTY,
    W_PFC_COMPOSITE,
    W_SA_PENALTY,
    W_TPSA,
    CO2ObjectiveScorer,
    _safe,
    tanimoto_diversity_scores,
    term_score,
)
from objectives.synthesizability import SynthesizabilityScorer  # noqa: E402
from mordred import Calculator  # noqa: E402
from mordred.EState import AtomTypeEState  # noqa: E402
from mordred.Polarizability import APol  # noqa: E402
from mordred.TopoPSA import TopoPSA  # noqa: E402
from transpharmer.generate import generate_yield  # noqa: E402
from transpharmer.SMILEValidifier import validate_smiles  # noqa: E402
from transpharmer.utils.io import load_config  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
DATA = OUT / "data"
CKPT_PRE = ROOT / "weights/guacamol_nc.pt"
# Set in main() from required CLI args.
CKPT_RL: Path
TRAIN_FINAL: Path

N_REPEAT = 2_000
N_TSNE_PROG = 500  # per-checkpoint sample size for --tsne-progression
N_CONFORMERS = 2  # match RL / reward.py CO2ObjectiveScorer
TEMPERATURE = 1.0
RANDOM_SEED = 42
TSNE_GIF_MS = 450

ALLOWED_ATOMS = {"C", "H", "N", "O", "F", "Cl", "Br", "I", "S"}

COLORS = {"pretrained": "#4C78A8", "rl": "#F58518"}


def _rl_tag(ckpt: Path) -> str:
    m = re.search(r"ep(\d+)", ckpt.name)
    return f"rl_ep{m.group(1)}" if m else "rl"


def _resolve_existing(path: Path, kind: str) -> Path:
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{kind} not found: {p}")
    return p


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def _release_cuda() -> None:
    """Drop GPU tensors left by generate_yield so sklearn/UMAP don't OpenMP-deadlock."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def _fit_tsne(Xp: np.ndarray) -> np.ndarray:
    """t-SNE with a single BLAS/OpenMP worker (avoids hang after CUDA sampling)."""
    print(f"Fitting t-SNE on {Xp.shape[0]} points...", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, Xp.shape[0] // 4)),
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
        n_jobs=1,
    )
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            emb = tsne.fit_transform(Xp)
    except ImportError:
        emb = tsne.fit_transform(Xp)
    print("t-SNE done.", flush=True)
    return emb


def _fit_umap(Xp: np.ndarray) -> np.ndarray:
    """UMAP with n_jobs=1 (same CUDA/OpenMP hang avoidance as t-SNE)."""
    import umap

    print(f"Fitting UMAP on {Xp.shape[0]} points...", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(5, Xp.shape[0] // 10)),
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            emb = reducer.fit_transform(Xp)
    except ImportError:
        emb = reducer.fit_transform(Xp)
    print("UMAP done.", flush=True)
    return emb


def sample_model(
    ckpt: Path,
    run_name: str,
    *,
    n_repeat: int | None = None,
    data_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return (all_raw_smiles, unique_valid_canonical)."""
    n = N_REPEAT if n_repeat is None else int(n_repeat)
    out_dir = DATA if data_dir is None else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{run_name}_raw.csv"
    valid_path = out_dir / f"{run_name}_valid.csv"
    if valid_path.exists() and raw_path.exists():
        raw = pd.read_csv(raw_path)["smiles"].astype(str).tolist()
        valid = pd.read_csv(valid_path)["SMILES"].astype(str).tolist()
        if len(raw) == n:
            print(f"Cached {run_name}: {len(raw)} raw, {len(valid)} valid")
            return raw, valid
        print(f"Stale cache {run_name}: {len(raw)} raw != n_repeat={n}; regenerating")

    cfg = load_config(str(ROOT / "generate.yaml"))
    cfg.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.RANDOM_SEED = RANDOM_SEED
    cfg.GENERATE.CKPT_PATH = str(ckpt)
    cfg.GENERATE.N_REPEAT = n
    cfg.GENERATE.BATCH_SIZE = min(256, n)
    cfg.GENERATE.TEMPERATURE = TEMPERATURE
    # ponytail: N_CPU=1 avoids CUDA-fork + OpenMP deadlocks in mapper Pool after sampling
    cfg.GENERATE.N_CPU = 1
    cfg.GENERATE.OUTPUT = str(raw_path)
    if raw_path.exists():
        raw_path.unlink()

    all_raw: list[str] = []
    valid: list[str] = []
    seen: set[str] = set()
    gen = generate_yield(cfg)
    try:
        for batch in gen:
            all_raw.extend(batch)
            for smi in validate_smiles(batch):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                can = Chem.MolToSmiles(mol)
                if can in seen:
                    continue
                seen.add(can)
                valid.append(can)
    finally:
        # Drop generator frame holding the GPT on GPU before sklearn runs.
        del gen
        _release_cuda()

    pd.DataFrame({"smiles": all_raw}).to_csv(raw_path, index=False)
    pd.DataFrame({"SMILES": valid}).to_csv(valid_path, index=False)
    print(f"Generated {run_name}: {len(all_raw)} raw, {len(valid)} unique valid")
    return all_raw, valid


def _discover_periodic_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return (episode, path) for periodic_*_epN.pt, sorted by episode."""
    pat = re.compile(r"^periodic_.+_ep(\d+)\.pt$")
    found: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("periodic_*_ep*.pt"):
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return found


def score_components(smiles_list: list[str], source: str) -> pd.DataFrame:
    """Score with the same sigmoid (0,1)×mw×sa×consec-rot reward used in training."""
    scorer = CO2ObjectiveScorer(
        random_seed=RANDOM_SEED,
        n_conformers=N_CONFORMERS,
        use_scaffold_filter=False,
        include_pfc=True,
        use_sigmoid_reward=True,
        n_workers=0,
    )
    sa_scorer = SynthesizabilityScorer()
    calc = Calculator(
        [
            APol(),
            TopoPSA(),
            AtomTypeEState("sum", "aaN"),
            AtomTypeEState("sum", "dNH"),
            AtomTypeEState("sum", "sNH2"),
            AtomTypeEState("sum", "ssNH"),
            AtomTypeEState("sum", "sOH"),
        ],
        ignore_3D=True,
    )
    diversity = tanimoto_diversity_scores(smiles_list)
    w_phys = (
        W_APOL
        + W_TPSA
        + W_BASICITY
        + W_LOGP
        + W_HALL_KIER_ALPHA
        + W_PFC_COMPOSITE
    )
    w_tot = w_phys + W_DIVERSE
    gate_ok_cache: dict[str, dict] = {}
    rows: list[dict] = []
    unique_scored = 0

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        div_raw = float(diversity[idx])
        div_u = term_score(div_raw, "diversity", use_sigmoid=True)
        row: dict = {
            "SMILES": smi,
            "source": source,
            "valid": mol is not None,
            "diversity_raw": div_raw,
            "diversity_term": div_u,
        }
        if mol is None:
            row.update(
                {
                    "gate_ok": False,
                    "reward": np.nan,
                    "apol": np.nan,
                    "TopoPSA": np.nan,
                    "basicity": np.nan,
                    "LogP": np.nan,
                    "HallKierAlpha": np.nan,
                    "PFC_composite": np.nan,
                    "PFC_term": np.nan,
                    "physchem_score": np.nan,
                    "pfc_only_score": np.nan,
                    "MW": np.nan,
                    "mw_scale": np.nan,
                    "sa_scale": np.nan,
                    "consec_rot": np.nan,
                    "consec_rot_scale": np.nan,
                    "n_oh": np.nan,
                    "n_amine": np.nan,
                    "oh_scale": np.nan,
                    "oh_excess": np.nan,
                }
            )
            rows.append(row)
            continue

        # Match training gate: formal charge + allowed atoms (MW is soft via mw_scale).
        mw = float(Descriptors.MolWt(mol))
        gate = Chem.GetFormalCharge(mol) == 0 and all(
            a.GetSymbol() in ALLOWED_ATOMS for a in mol.GetAtoms()
        )
        row["MW"] = mw
        row["gate_ok"] = gate
        if not gate:
            row["reward"] = np.nan
            row["mw_scale"] = np.nan
            row["sa_scale"] = np.nan
            row["consec_rot"] = np.nan
            row["consec_rot_scale"] = np.nan
            row["n_oh"] = np.nan
            row["n_amine"] = np.nan
            row["oh_scale"] = np.nan
            row["oh_excess"] = np.nan
            rows.append(row)
            continue

        scale = float(
            mw_scale(
                mw,
                lo=MW_SCALE_LO,
                hi=MW_SCALE_HI,
                inner_lo=MW_FLAT_LO,
                inner_hi=MW_FLAT_HI,
            )
        )
        can = Chem.MolToSmiles(mol)
        try:
            sa = float(sa_scorer.score(can))
        except Exception:
            sa = SA_SCALE_ZERO_HI
        sa_f = sa_scale(sa, full_hi=SA_SCALE_FULL_HI, zero_hi=SA_SCALE_ZERO_HI)
        crot = max_consecutive_rotatable_bonds(mol)
        crot_f = consec_rot_scale(
            float(crot), full_hi=CONSEC_ROT_FULL_HI, zero_hi=CONSEC_ROT_ZERO_HI
        )
        n_oh = count_oh(mol)
        n_amine = count_amine_n(mol)
        oh_f = oh_scale(
            float(n_oh), full_hi=OH_SCALE_FULL_HI, zero_hi=OH_SCALE_ZERO_HI
        )
        oh_ex = amine_justified_oh_excess(
            float(n_oh), float(n_amine), free=OH_AMINE_FREE
        )
        penalty = (
            W_MW_PENALTY
            * mw_excess(
                mw,
                lo=MW_SCALE_LO,
                hi=MW_SCALE_HI,
                inner_lo=MW_FLAT_LO,
                inner_hi=MW_FLAT_HI,
            )
            + W_SA_PENALTY
            * high_end_excess(sa, full_hi=SA_SCALE_FULL_HI, zero_hi=SA_SCALE_ZERO_HI)
            + W_CONSEC_ROT_PENALTY
            * high_end_excess(
                float(crot), full_hi=CONSEC_ROT_FULL_HI, zero_hi=CONSEC_ROT_ZERO_HI
            )
            + W_OH_PENALTY * oh_ex
        )
        cached = gate_ok_cache.get(can)
        if cached is None:
            result = calc(mol)
            logp = float(Descriptors.MolLogP(mol))
            alpha = float(Descriptors.HallKierAlpha(mol))
            apol = float(result["apol"])
            tpsa = float(result["TopoPSA(NO)"])
            basicity = (
                _safe(result["SaaN"])
                + _safe(result["SdNH"])
                + _safe(result["SsNH2"])
                + _safe(result["SssNH"])
                + _safe(result["SsOH"])
            )
            try:
                pfc = scorer.pfc.compute(smi)
                pfc_raw = float(pfc["composite"])
            except Exception:
                pfc_raw = float("nan")

            apol_t = term_score(apol, "apol", use_sigmoid=True)
            tpsa_t = term_score(tpsa, "TopoPSA", use_sigmoid=True)
            bas_t = term_score(basicity, "basicity", use_sigmoid=True)
            logp_t = term_score(logp, "LogP", use_sigmoid=True)
            alpha_t = term_score(alpha, "HallKierAlpha", use_sigmoid=True)
            pfc_term = (
                term_score(pfc_raw, "PFC_composite", use_sigmoid=True)
                if math.isfinite(pfc_raw)
                else 0.0
            )
            # Weighted mean of unit terms ∈ (0, 1); LogP prefers low → (1 - sig).
            physchem_u = (
                W_APOL * apol_t
                + W_TPSA * tpsa_t
                + W_BASICITY * bas_t
                + W_LOGP * (1.0 - logp_t)
                + W_HALL_KIER_ALPHA * alpha_t
                + W_PFC_COMPOSITE * pfc_term
            ) / w_phys
            cached = {
                "apol": apol,
                "TopoPSA": tpsa,
                "basicity": basicity,
                "LogP": logp,
                "HallKierAlpha": alpha,
                "PFC_composite": pfc_raw,
                "PFC_term": pfc_term,
                "apol_term": apol_t,
                "TopoPSA_term": tpsa_t,
                "basicity_term": bas_t,
                "LogP_term": logp_t,
                "HallKierAlpha_term": alpha_t,
                "physchem_u": physchem_u,
                "pfc_only_score": W_PFC_COMPOSITE * pfc_term,
            }
            gate_ok_cache[can] = cached
            unique_scored += 1

        physchem_u = float(cached["physchem_u"])
        reward_u = (physchem_u * w_phys + div_u * W_DIVERSE) / w_tot
        row.update(cached)
        row["mw_scale"] = scale
        row["sa_scale"] = sa_f
        row["consec_rot"] = crot
        row["consec_rot_scale"] = crot_f
        row["n_oh"] = n_oh
        row["n_amine"] = n_amine
        row["oh_scale"] = oh_f
        row["oh_excess"] = oh_ex
        row["soft_penalty"] = penalty
        row["physchem_score"] = physchem_u - penalty
        row["reward"] = float(reward_u - penalty)
        rows.append(row)
        if (idx + 1) % 100 == 0 or idx + 1 == len(smiles_list):
            print(
                f"  scored {idx + 1}/{len(smiles_list)} ({source})"
                f" unique_gate_ok={unique_scored}",
                flush=True,
            )

    return pd.DataFrame(rows)


def load_or_score() -> pd.DataFrame:
    # Filename includes n_conformers so a 5-conf cache is not reused after N_CONFORMERS=2.
    scored_path = DATA / f"scored_components_nconf{N_CONFORMERS}_sigmoid.csv"
    rl_tag = _rl_tag(CKPT_RL)
    rl_raw_path = DATA / f"{rl_tag}_raw.csv"
    # Cache is only valid for this RL checkpoint tag (epN in the filename).
    if scored_path.exists() and rl_raw_path.exists():
        df = pd.read_csv(scored_path)
        # Expect one row per generated SMILES per source
        counts = df.groupby("source").size()
        if len(df) == 2 * N_REPEAT and counts.min() == N_REPEAT:
            print(f"Cached scored components: {len(df)} ({scored_path.name}, {rl_tag})")
            return df
        print(f"Stale scored cache: {dict(counts)} (need {N_REPEAT} each); rescoring")
    elif scored_path.exists() and not rl_raw_path.exists():
        print(f"Scored cache is for a different checkpoint (missing {rl_raw_path.name}); regenerating")

    pre_raw, _ = sample_model(CKPT_PRE, "pretrained")
    rl_raw, _ = sample_model(CKPT_RL, rl_tag)
    _release_cuda()
    print(f"Scoring all generated: pretrained={len(pre_raw)} rl={len(rl_raw)} (PFC n_conformers={N_CONFORMERS})...")
    df = pd.concat(
        [score_components(pre_raw, "pretrained"), score_components(rl_raw, "rl")],
        ignore_index=True,
    )
    df.to_csv(scored_path, index=False)
    print(f"Wrote scored cache {scored_path.name} ({len(df)} rows); building figures...", flush=True)
    return df


def fig01_learning_curve() -> None:
    """Copy training curve image into figures/."""
    shutil.copy2(TRAIN_FINAL, FIG / "learning_curve.png")
    print(f"Wrote {FIG / 'learning_curve.png'} (from {TRAIN_FINAL})")


def fig02_baseline_comparison(df: pd.DataFrame) -> None:
    def _violin(ax, frame: pd.DataFrame, col: str, title: str) -> None:
        data = [frame.loc[frame.source == s, col].dropna().values for s in ("pretrained", "rl")]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor([COLORS["pretrained"], COLORS["rl"]][i])
            body.set_alpha(0.7)
        ax.set_xticks([0, 1], ["Pretrained", "RL"])
        ax.set_ylabel(col)
        ax.set_title(title)
        for i, (s, c) in enumerate(zip(("pretrained", "rl"), (COLORS["pretrained"], COLORS["rl"]))):
            vals = frame.loc[frame.source == s, col].dropna()
            if len(vals) == 0:
                continue
            ax.annotate(
                f"μ={vals.mean():.3f}",
                xy=(i, vals.mean()),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=c,
            )

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _violin(ax, df, "reward", "CO2 reward")
    fig.suptitle("Distribution shift: all generated molecules", y=1.02)
    _save(fig, "baseline_reward.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for s, c, lab in (
        ("pretrained", COLORS["pretrained"], "Pretrained"),
        ("rl", COLORS["rl"], "RL"),
    ):
        vals = np.sort(df.loc[df.source == s, "reward"].dropna().values)
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=c, lw=2, label=lab)
    ax.set_xlabel("reward")
    ax.set_ylabel("ECDF")
    ax.set_title("CO2 reward")
    ax.legend(frameon=False)
    fig.suptitle("ECDF: all generated molecules", y=1.02)
    _save(fig, "baseline_ecdf.png")


def fig03_property_space(df: pd.DataFrame) -> None:
    # reward/diversity exist for all; physchem/PFC are NaN on gate fails (dropped in hist)
    props = [
        ("reward", "Total CO2 reward", None),
        ("diversity_raw", "Tanimoto diversity", None),
        ("PFC_composite", "PFC composite", RANGES["PFC_composite"]),
        ("TopoPSA", "TopoPSA", RANGES["TopoPSA"]),
        ("apol", "APol", RANGES["apol"]),
        ("LogP", "LogP", RANGES["LogP"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (col, title, rng) in zip(axes.ravel(), props):
        for s, c, lab in (
            ("pretrained", COLORS["pretrained"], "Pretrained"),
            ("rl", COLORS["rl"], "RL"),
        ):
            vals = df.loc[df.source == s, col].dropna()
            ax.hist(vals, bins=30, density=True, alpha=0.55, color=c, label=lab)
        if rng is not None:
            lo, hi = rng
            ax.axvline(lo, color="gray", ls="--", lw=1, label="sigmoid lo/hi" if col == "PFC_composite" else None)
            ax.axvline(hi, color="gray", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel(col)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("CO2 property-space shift (all generated; physchem where scored)", y=1.01)
    _save(fig, "property_space.png")


def fig04_top_hits(df: pd.DataFrame, k: int = 12) -> None:
    rl = df[df.source == "rl"]
    path = FIG / "top_hits_gallery.png"
    top = _render_top_hits_gallery(rl, path, k=k)
    if top is None:
        return
    top[["SMILES", "reward", "PFC_composite", "LogP", "TopoPSA", "apol", "basicity"]].to_csv(
        DATA / "top_hits.csv", index=False
    )


def _render_top_hits_gallery(
    frame: pd.DataFrame,
    path: Path,
    *,
    k: int = 12,
    title: str | None = None,
) -> pd.DataFrame | None:
    """Draw top-k by reward; return the top table or None if empty."""
    ranked = frame.dropna(subset=["reward"]).sort_values("reward", ascending=False)
    top = ranked.head(k)
    mols, legends = [], []
    for _, r in top.iterrows():
        mol = Chem.MolFromSmiles(r.SMILES)
        if mol is None:
            continue
        mols.append(mol)
        pfc = r.PFC_composite
        pfc_s = f"{pfc:.3f}" if pd.notna(pfc) else "n/a"
        legends.append(f"R={r.reward:.3f}\nPFC={pfc_s}")
    if not mols:
        return None
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 280), legends=legends)
    if title:
        from PIL import Image, ImageDraw, ImageFont

        grid = img.convert("RGB")
        banner_h = 36
        canvas = Image.new("RGB", (grid.width, grid.height + banner_h), (255, 255, 255))
        canvas.paste(grid, (0, banner_h))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        draw.text((12, 8), title, fill=(30, 30, 30), font=font)
        canvas.save(str(path))
    else:
        img.save(str(path))
    print(f"Wrote {path}")
    return top


def _stitch_progression_gif(pngs: list[Path], gif_path: Path) -> None:
    from PIL import Image

    if not pngs:
        return
    images = [Image.open(p).convert("RGB") for p in pngs]
    # Match sizes (top-hit grids can differ slightly if mol count varies).
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    padded = []
    for im in images:
        if im.size == (w, h):
            padded.append(im)
            continue
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(im, (0, 0))
        padded.append(canvas)
        im.close()
    padded[0].save(
        gif_path,
        save_all=True,
        append_images=padded[1:],
        duration=TSNE_GIF_MS,
        loop=0,
    )
    for im in padded:
        im.close()
    print(f"Wrote {gif_path}")


def _morgan_fps(smiles: list[str]):
    fps = []
    keep = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        keep.append(i)
    return fps, keep


def _chemotype_fp_matrix(df: pd.DataFrame):
    """Return (sub_df, Xp) after Morgan FP → PCA, or (None, None) if too small."""
    frame = df.reset_index(drop=True)
    fps, keep = _morgan_fps(frame.SMILES.tolist())
    if len(fps) < 10:
        return None, None
    sub = frame.iloc[keep].reset_index(drop=True)
    X = np.zeros((len(fps), 2048), dtype=np.float32)
    for i, fp in enumerate(fps):
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    Xp = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit_transform(X)
    return sub, Xp


def _add_95pct_ellipse(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    color: str,
    label: str,
    zorder: float = 0.5,
) -> None:
    """Shade a 2D Gaussian 95% confidence ellipse (χ², 2 dof)."""
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        return
    pts = np.column_stack([x, y])
    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return
    # Stabilize near-singular clouds.
    cov = cov + np.eye(2) * 1e-9
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[0] <= 0 or vals[1] <= 0:
        return
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = float(np.sqrt(chi2.ppf(0.95, df=2)))
    width, height = 2.0 * scale * np.sqrt(vals)
    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor=color,
        alpha=0.18,
        linewidth=1.5,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(ell)


def _plot_chemotype(
    emb: np.ndarray,
    sub: pd.DataFrame,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    outfile: str,
) -> None:
    """Pretrained muted; RL plasma_r (light=low, dark=high) with reward-scaled alpha/zorder."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    m_pre = sub.source.values == "pretrained"
    m_rl = sub.source.values == "rl"

    _add_95pct_ellipse(
        ax,
        emb[m_pre, 0],
        emb[m_pre, 1],
        color=COLORS["pretrained"],
        label="Pretrained 95%",
        zorder=0.4,
    )
    _add_95pct_ellipse(
        ax,
        emb[m_rl, 0],
        emb[m_rl, 1],
        color=COLORS["rl"],
        label="RL 95%",
        zorder=0.5,
    )

    ax.scatter(
        emb[m_pre, 0],
        emb[m_pre, 1],
        c=sub.loc[m_pre, "reward"],
        cmap="Blues",
        s=28,
        alpha=0.4,
        marker="^",
        linewidths=0.35,
        edgecolors=COLORS["pretrained"],
        label="Pretrained",
        zorder=1,
    )

    rl_idx = np.flatnonzero(m_rl)
    rl_reward = sub.loc[m_rl, "reward"].to_numpy(dtype=float)
    # Drop gate-fail NaNs (otherwise min/max → nan and matplotlib clim defaults to ±0.1).
    finite = np.isfinite(rl_reward)
    rl_idx = rl_idx[finite]
    rl_reward = rl_reward[finite]
    order = np.argsort(rl_reward)  # low reward first → drawn under high
    rl_idx = rl_idx[order]
    rl_reward = rl_reward[order]
    rmin, rmax = float(np.min(rl_reward)), float(np.max(rl_reward))
    t = (rl_reward - rmin) / (rmax - rmin + 1e-12)
    cmap = plt.colormaps["plasma_r"]  # light (low) → dark (high)
    rgba = cmap(t)
    rgba[:, 3] = 0.15 + 0.80 * t  # low score → faint

    n_chunks = 12
    chunks = np.array_split(np.arange(len(rl_idx)), n_chunks)
    for ci, chunk in enumerate(chunks):
        ii = rl_idx[chunk]
        ax.scatter(
            emb[ii, 0],
            emb[ii, 1],
            c=rgba[chunk],
            s=44,
            marker="o",
            linewidths=0.55,
            edgecolors=COLORS["rl"],
            label="RL" if ci == n_chunks - 1 else None,
            zorder=2 + ci,
        )
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=rmin, vmax=rmax)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RL CO2 reward")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    _save(fig, outfile)


def fig06_chemotype_map(df: pd.DataFrame) -> None:
    sub, Xp = _chemotype_fp_matrix(df)
    if sub is None:
        return
    _release_cuda()
    emb_tsne = _fit_tsne(Xp)
    _plot_chemotype(
        emb_tsne,
        sub,
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        title="Chemotype map (all generated; Morgan FP → PCA → t-SNE)",
        outfile="chemotype_tsne.png",
    )

    emb_umap = _fit_umap(Xp)
    _plot_chemotype(
        emb_umap,
        sub,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title="Chemotype map (all generated; Morgan FP → PCA → UMAP)",
        outfile="chemotype_umap.png",
    )


def _load_or_build_tsne_prog_scores(
    periodics: list[tuple[int, Path]],
) -> pd.DataFrame:
    """Sample + score pretrained and each periodic into data/tsne_prog/ (cached)."""
    prog_dir = DATA / "tsne_prog"
    prog_dir.mkdir(parents=True, exist_ok=True)
    scored_path = prog_dir / "scored_sigmoid.csv"
    expected_sources = {"pretrained"} | {f"ep{ep}" for ep, _ in periodics}

    if scored_path.exists():
        df = pd.read_csv(scored_path)
        have = set(df["source"].astype(str).unique())
        counts = df.groupby("source").size()
        ok_counts = all(
            int(counts.get(s, 0)) == N_TSNE_PROG for s in expected_sources if s in have
        )
        if expected_sources <= have and ok_counts:
            print(
                f"Cached t-SNE progression scores: {len(df)} "
                f"({scored_path.name}, episodes={[e for e, _ in periodics]})"
            )
            return df.loc[df["source"].isin(expected_sources)].reset_index(drop=True)
        print(
            f"Stale t-SNE progression cache (have={sorted(have)}, "
            f"need={sorted(expected_sources)}); regenerating"
        )

    frames: list[pd.DataFrame] = []
    pre_raw, _ = sample_model(
        CKPT_PRE,
        "pretrained",
        n_repeat=N_TSNE_PROG,
        data_dir=prog_dir,
    )
    print(f"Scoring t-SNE progression pretrained ({len(pre_raw)})...")
    frames.append(score_components(pre_raw, "pretrained"))

    for ep, ckpt in periodics:
        tag = f"ep{ep}"
        raw, _ = sample_model(
            ckpt,
            f"tsne_{tag}",
            n_repeat=N_TSNE_PROG,
            data_dir=prog_dir,
        )
        print(f"Scoring t-SNE progression {tag} ({len(raw)})...")
        frames.append(score_components(raw, tag))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(scored_path, index=False)
    return df


def _plot_tsne_prog_frame(
    emb: np.ndarray,
    sub: pd.DataFrame,
    *,
    ep: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    rmin: float,
    rmax: float,
    out_path: Path,
) -> None:
    """One progression frame: pretrained + this episode's RL in shared embedding."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    m_pre = sub.source.values == "pretrained"
    src = f"ep{ep}"
    m_rl = sub.source.values == src

    _add_95pct_ellipse(
        ax,
        emb[m_pre, 0],
        emb[m_pre, 1],
        color=COLORS["pretrained"],
        label="Pretrained 95%",
        zorder=0.4,
    )
    _add_95pct_ellipse(
        ax,
        emb[m_rl, 0],
        emb[m_rl, 1],
        color=COLORS["rl"],
        label="RL 95%",
        zorder=0.5,
    )

    ax.scatter(
        emb[m_pre, 0],
        emb[m_pre, 1],
        c=sub.loc[m_pre, "reward"],
        cmap="Blues",
        s=28,
        alpha=0.35,
        marker="^",
        linewidths=0.35,
        edgecolors=COLORS["pretrained"],
        label="Pretrained",
        zorder=1,
    )

    rl_idx = np.flatnonzero(m_rl)
    rl_reward = sub.loc[m_rl, "reward"].to_numpy(dtype=float)
    finite = np.isfinite(rl_reward)
    rl_idx = rl_idx[finite]
    rl_reward = rl_reward[finite]
    cmap = plt.colormaps["plasma_r"]
    if len(rl_idx) > 0:
        order = np.argsort(rl_reward)
        rl_idx = rl_idx[order]
        rl_reward = rl_reward[order]
        t = (rl_reward - rmin) / (rmax - rmin + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        rgba = cmap(t)
        rgba[:, 3] = 0.15 + 0.80 * t
        n_chunks = min(12, max(1, len(rl_idx)))
        chunks = np.array_split(np.arange(len(rl_idx)), n_chunks)
        for ci, chunk in enumerate(chunks):
            ii = rl_idx[chunk]
            ax.scatter(
                emb[ii, 0],
                emb[ii, 1],
                c=rgba[chunk],
                s=44,
                marker="o",
                linewidths=0.55,
                edgecolors=COLORS["rl"],
                label="RL" if ci == len(chunks) - 1 else None,
                zorder=2 + ci,
            )

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=rmin, vmax=rmax)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RL CO2 reward")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Chemotype progression — episode {ep}")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_tsne_progression() -> None:
    """Opt-in: periodics → shared t-SNE frames + top-hits galleries + GIFs."""
    periodics = _discover_periodic_checkpoints(CKPT_RL.parent)
    if not periodics:
        print(
            f"No periodic_*_ep*.pt under {CKPT_RL.parent}; skipping --tsne-progression",
            flush=True,
        )
        return

    print(
        f"Checkpoint progression: {len(periodics)} periodics "
        f"(ep {[e for e, _ in periodics]})",
        flush=True,
    )
    df = _load_or_build_tsne_prog_scores(periodics)
    fig_top_hits_progression(df, periodics)
    _fig_tsne_progression_frames(df, periodics)


def fig_top_hits_progression(
    df: pd.DataFrame,
    periodics: list[tuple[int, Path]],
    *,
    k: int = 12,
) -> None:
    """Per-episode top-k galleries under top_hits/ + progression.gif."""
    hits_dir = OUT / "top_hits"
    hits_dir.mkdir(parents=True, exist_ok=True)
    print(f"Top-hits progression → {hits_dir}", flush=True)
    pngs: list[Path] = []
    for ep, _ in periodics:
        sub = df[df.source == f"ep{ep}"]
        out_path = hits_dir / f"ep{ep:04d}.png"
        top = _render_top_hits_gallery(
            sub, out_path, k=k, title=f"Top hits — episode {ep}"
        )
        if top is None:
            print(f"  skip ep{ep}: no scored hits", flush=True)
            continue
        top.to_csv(hits_dir / f"ep{ep:04d}_hits.csv", index=False)
        pngs.append(out_path)
    if pngs:
        _stitch_progression_gif(pngs, hits_dir / "progression.gif")


def _fig_tsne_progression_frames(
    df: pd.DataFrame,
    periodics: list[tuple[int, Path]],
) -> None:
    """Joint t-SNE frames under tsne/ + progression.gif."""
    tsne_dir = OUT / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    print(f"t-SNE progression → {tsne_dir}", flush=True)

    sub, Xp = _chemotype_fp_matrix(df)
    if sub is None:
        print("t-SNE progression: too few fingerprints; skipping", flush=True)
        return

    _release_cuda()
    emb = _fit_tsne(Xp)

    pad = 0.05
    x0, x1 = float(emb[:, 0].min()), float(emb[:, 0].max())
    y0, y1 = float(emb[:, 1].min()), float(emb[:, 1].max())
    dx, dy = x1 - x0, y1 - y0
    xlim = (x0 - pad * dx, x1 + pad * dx)
    ylim = (y0 - pad * dy, y1 + pad * dy)

    ep_mask = sub["source"].astype(str).str.match(r"^ep\d+$")
    rl_rewards = sub.loc[ep_mask, "reward"].to_numpy(dtype=float)
    rl_rewards = rl_rewards[np.isfinite(rl_rewards)]
    if rl_rewards.size == 0:
        print("t-SNE progression: no finite RL rewards; skipping", flush=True)
        return
    rmin, rmax = float(np.min(rl_rewards)), float(np.max(rl_rewards))
    if rmin == rmax:
        rmax = rmin + 1e-6

    pngs: list[Path] = []
    for ep, _ in periodics:
        out_path = tsne_dir / f"ep{ep:04d}.png"
        _plot_tsne_prog_frame(
            emb,
            sub,
            ep=ep,
            xlim=xlim,
            ylim=ylim,
            rmin=rmin,
            rmax=rmax,
            out_path=out_path,
        )
        pngs.append(out_path)

    _stitch_progression_gif(pngs, tsne_dir / "progression.gif")


def fig07_validity(df: pd.DataFrame) -> None:
    rl_tag = _rl_tag(CKPT_RL)
    rows = []
    for name, raw_file in (
        ("pretrained", DATA / "pretrained_raw.csv"),
        ("rl", DATA / f"{rl_tag}_raw.csv"),
    ):
        if not raw_file.exists():
            continue
        raw = pd.read_csv(raw_file)["smiles"].astype(str).tolist()
        n = len(raw)
        valid_smi = []
        seen = set()
        n_valid = 0
        for smi in raw:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n_valid += 1
            can = Chem.MolToSmiles(mol)
            if can not in seen:
                seen.add(can)
                valid_smi.append(can)
        sub = df[df.source == ("rl" if name == "rl" else "pretrained")]
        # Remap: our scored df uses source pretrained/rl
        src = "rl" if name == "rl" else "pretrained"
        sub = df[df.source == src]
        n_gate = int((sub["gate_ok"] == True).sum())  # noqa: E712
        rows.append(
            {
                "source": name,
                "n_generated": n,
                "pct_valid": 100.0 * n_valid / max(n, 1),
                "pct_unique_of_valid": 100.0 * len(seen) / max(n_valid, 1),
                "pct_gate_of_unique": 100.0 * n_gate / max(len(sub), 1),
            }
        )
    if not rows:
        return
    stats = pd.DataFrame(rows)
    stats.to_csv(DATA / "validity_stats.csv", index=False)

    metrics = ["pct_valid", "pct_unique_of_valid", "pct_gate_of_unique"]
    labels = ["Valid SMILES %", "Unique / valid %", "Gate-pass / scored %"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sources = [s for s in ("pretrained", "rl") if (stats.source == s).any()]
    if not sources:
        return
    for i, src in enumerate(sources):
        c = COLORS[src]
        part = stats.loc[stats.source == src]
        vals = [float(part[m].iloc[0]) for m in metrics]
        ax.bar(
            x + (i - 0.5 * (len(sources) - 1)) * width,
            vals,
            width,
            color=c,
            label="RL" if src == "rl" else src.capitalize(),
        )
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_title("Generation quality: validity, uniqueness, CO2 gates")
    ax.legend(frameon=False)
    _save(fig, "validity_uniqueness_gates.png")


def fig08_physchem_overlay(df: pd.DataFrame) -> None:
    """Pretrained vs RL physchem distributions (no external PCI reference)."""
    props = ["LogP", "TopoPSA", "apol", "basicity", "HallKierAlpha"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, props):
        for lab, c, src in (
            ("Pretrained", COLORS["pretrained"], "pretrained"),
            ("RL", COLORS["rl"], "rl"),
        ):
            series = df.loc[df.source == src, col].dropna()
            if len(series) == 0:
                continue
            ax.hist(series, bins=30, density=True, alpha=0.45, color=c, label=lab)
        ax.set_title(col)
    ax = axes[5]
    for lab, c, src in (
        ("Pretrained", COLORS["pretrained"], "pretrained"),
        ("RL", COLORS["rl"], "rl"),
    ):
        ax.hist(df.loc[df.source == src, "reward"].dropna(), bins=30, density=True, alpha=0.5, color=c, label=lab)
    ax.set_title("Total CO2 reward (all generated)")
    ax.legend(frameon=False, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Physchem shift: pretrained vs RL (all generated)", y=1.01)
    _save(fig, "physchem_overlay.png")


def fig_summary_table(df: pd.DataFrame) -> None:
    rows = []
    for src in ("pretrained", "rl"):
        sub = df[df.source == src]
        rows.append(
            {
                "source": src,
                "n": len(sub),
                "gate_frac": float((sub.gate_ok == True).mean()),  # noqa: E712
                "reward_mean": sub.reward.mean(),
                "reward_median": sub.reward.median(),
                "PFC_mean": sub.PFC_composite.mean(),
                "diversity_mean": sub.diversity_raw.mean(),
                "LogP_mean": sub.LogP.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "summary_stats.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    num = summary.select_dtypes(include=[np.number])
    tbl = ax.table(
        cellText=np.round(num.values, 3),
        rowLabels=summary["source"].tolist(),
        colLabels=list(num.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax.set_title("Summary statistics (all generated molecules)", pad=12)
    _save(fig, "summary_stats.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CO2 figures (default under publication/co2/; override with --out-dir)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="RL checkpoint .pt (e.g. outputs/.../checkpoints/best_default_ep450.pt)",
    )
    parser.add_argument(
        "--train-graph",
        type=Path,
        required=True,
        help="Training curve image (e.g. outputs/.../graphs/final.png)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root for figures/ and data/ (default: publication/co2/)",
    )
    parser.add_argument(
        "--tsne-progression",
        action="store_true",
        help=(
            "Also sample each periodic_*_ep*.pt (N=500), then write "
            "tsne/ and top_hits/ epXXXX.png + progression.gif"
        ),
    )
    args = parser.parse_args()

    global CKPT_RL, TRAIN_FINAL, FIG, DATA, OUT
    CKPT_RL = _resolve_existing(args.checkpoint, "checkpoint")
    TRAIN_FINAL = _resolve_existing(args.train_graph, "train graph")
    if args.out_dir is not None:
        OUT = args.out_dir.expanduser().resolve()
        FIG = OUT / "figures"
        DATA = OUT / "data"

    os.chdir(ROOT)
    os.environ.setdefault("MGW_OBJECTIVE", "co2")
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    _style()

    print(f"RL checkpoint: {CKPT_RL}")
    print(f"Train graph:   {TRAIN_FINAL}")
    print(f"Output dir:    {OUT}")

    fig01_learning_curve()
    df = load_or_score()
    fig_summary_table(df)
    fig02_baseline_comparison(df)
    fig03_property_space(df)
    fig04_top_hits(df)
    fig06_chemotype_map(df)
    fig07_validity(df)
    fig08_physchem_overlay(df)
    if args.tsne_progression:
        fig_tsne_progression()
    print("Done.")


if __name__ == "__main__":
    main()
