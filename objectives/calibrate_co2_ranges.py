#!/usr/bin/env python3
"""Compare GuacaMol / AxelRolov boxes vs current GuacaMol∪AxelRolov RANGES.

Also prints intensive (col/n_HA) p10/p90 for apol, TopoPSA, basicity.

Run: PYTHONPATH=. uv run python objectives/calibrate_co2_ranges.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from objectives.co2_reward import RANGES, sigmoid_to_unit

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
GUA_CSV = ROOT / "publication/co2/data/scored_components_nconf2.csv"
AXEL_CSV = ROOT / "publication/co2/data/axelrolov_scored_nconf2.csv"

# Cached p10/p90 from each corpus (see axelrolov_scored_nconf2.csv + GuacaMol gated).
GUACAMOL_RANGES = {
    "apol": (38.3, 68.8),
    "TopoPSA": (37.0, 110.2),
    "basicity": (0.0, 20.4),
    "LogP": (1.33, 5.29),
    "HallKierAlpha": (-3.61, -1.30),
    "PFC_composite": (0.41, 0.74),
    "diversity": (0.70, 0.95),
}

AXEL_RANGES = {
    "apol": (21.9, 39.8),
    "TopoPSA": (21.7, 46.9),
    "basicity": (0.0, 18.5),
    "LogP": (-0.63, 0.87),
    "HallKierAlpha": (-0.16, -0.08),
    "PFC_composite": (0.35, 0.51),
    "diversity": (0.0, 1.0),
}

COL_MAP = {
    "apol": "apol",
    "TopoPSA": "TopoPSA",
    "basicity": "basicity",
    "LogP": "LogP",
    "HallKierAlpha": "HallKierAlpha",
    "PFC_composite": "PFC_composite",
    "diversity": "diversity_raw",
}

INTENSIVE_KEYS = ("apol", "TopoPSA", "basicity")


def _n_ha_from_smiles(smi: str) -> float:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return float("nan")
    return float(max(int(mol.GetNumHeavyAtoms()), 1))


def _safe_sig(x: float, lo: float, hi: float) -> float:
    try:
        return float(sigmoid_to_unit(float(x), float(lo), float(hi)))
    except OverflowError:
        t = (float(x) - float(lo)) / max(float(hi) - float(lo), 1e-9)
        return 0.0 if t < 0.5 else 1.0


def _series_stats(s: pd.Series) -> tuple[float, float, float, float]:
    a = s.dropna().to_numpy(dtype=float)
    return (
        float(np.mean(a)),
        float(np.std(a, ddof=0)),
        float(np.quantile(a, 0.1)),
        float(np.quantile(a, 0.9)),
    )


def main() -> None:
    df = pd.read_csv(GUA_CSV)
    g = df[(df["source"] == "pretrained") & (df["gate_ok"] == True)].copy()  # noqa: E712
    print(f"pretrained gated n={len(g)}")
    axel = pd.read_csv(AXEL_CSV)
    print(f"axelrolov scored n={len(axel)}")

    print("\npercentiles + mean sigmoid (GuacaMol / AxelRolov / union RANGES):")
    for key, col in COL_MAP.items():
        s = g[col].dropna()
        p10, p50, p90 = s.quantile(0.1), s.median(), s.quantile(0.9)
        gua_m = float(s.map(lambda x: _safe_sig(x, *GUACAMOL_RANGES[key])).mean())
        ax_m = float(s.map(lambda x: _safe_sig(x, *AXEL_RANGES[key])).mean())
        uni_m = float(s.map(lambda x: _safe_sig(x, *RANGES[key])).mean())
        print(
            f"  {key}: p10={p10:.4g} p50={p50:.4g} p90={p90:.4g} "
            f"sig_gua={gua_m:.3f} sig_axel={ax_m:.3f} sig_union={uni_m:.3f} "
            f"ranges={RANGES[key]}"
        )

    g = g.dropna(subset=["SMILES", "apol", "TopoPSA", "basicity"]).copy()
    g["n_ha"] = g["SMILES"].map(_n_ha_from_smiles)
    g = g.dropna(subset=["n_ha"])
    ax = axel.dropna(subset=["n_ha", "apol", "TopoPSA", "basicity"]).copy()
    ax["n_ha"] = ax["n_ha"].clip(lower=1)

    print("\nintensive (col/n_HA) GuacaMol-prior gated mean/std + union p10/p90:")
    for col in INTENSIVE_KEYS:
        mu, sd, g10, g90 = _series_stats(g[col] / g["n_ha"])
        _, _, a10, a90 = _series_stats(ax[col] / ax["n_ha"])
        print(
            f"  {col}/n_ha: REF=({mu:.6g}, {sd:.6g}) "
            f"union_p10={min(g10, a10):.4g} union_p90={max(g90, a90):.4g} "
            f"ranges={RANGES[col]}"
        )

    mid_apol = sum(RANGES["apol"]) / 2.0
    mid_logp = sum(RANGES["LogP"]) / 2.0
    apol_mid = sigmoid_to_unit(mid_apol, *RANGES["apol"])
    logp_mid = sigmoid_to_unit(mid_logp, *RANGES["LogP"])
    assert 0.3 <= apol_mid <= 0.7, f"APol midpoint sigmoid {apol_mid} not mid-slope"
    assert 0.3 <= logp_mid <= 0.7, f"LogP midpoint sigmoid {logp_mid} not mid-slope"
    print(f"\nOK: union midpoint APol sigmoid={apol_mid:.3f}, LogP sigmoid={logp_mid:.3f}")


if __name__ == "__main__":
    main()
