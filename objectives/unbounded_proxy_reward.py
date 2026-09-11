"""Single-descriptor unbounded proxies for reward-hacking demos.

Policy maximizes one raw RDKit descriptor (or MolLogP+MolWt), minus an optional
soft MW penalty that grows exponentially above a threshold (default 400 Da).
Gate is parse + atom_check only. Policy signal = batch z-score of the penalized
score among gated mols.

Select with::

    MGW_OBJECTIVE=unbounded_proxy
    MGW_PROXY=mollogp|molwt|heavyatomcount|bertzct|numrotatablebonds|tpsa|mollogp_molwt

MW soft penalty (not a hard gate)::

    MGW_MW_PENALTY_START=400   # Da; <=0 disables
    MGW_MW_PENALTY_TAU=50      # length scale; penalty = expm1((mw-start)/tau)

Companion descriptors are always logged into RewardBatch slots for diagnostics
(even when they are not the training proxy):
  raw_ba=MolLogP, raw_physchem=MolWt, mw_scale=HeavyAtomCount, raw_div=BertzCT
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from transpharmer.SMILEValidifier import atom_check

INVALID_PENALTY = -1.0

PROXY_NAMES = (
    "mollogp",
    "molwt",
    "heavyatomcount",
    "bertzct",
    "numrotatablebonds",
    "tpsa",
    "mollogp_molwt",
)

# Soft oversize penalty: 0 at START, ~1.7 at START+TAU, ~6.4 at START+2·TAU.
DEFAULT_MW_PENALTY_START = 400.0
DEFAULT_MW_PENALTY_TAU = 50.0


@dataclass(frozen=True)
class UnboundedProxyBatch:
    policy: list[float]
    raw_total: list[float]
    quality: list[float]
    gate_ok: list[bool]
    mollogp: list[float]
    molwt: list[float]
    heavyatomcount: list[float]
    bertzct: list[float]


def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + 1e-8)


def normalize_proxy_name(name: str | None) -> str:
    key = (name or "mollogp").strip().lower().replace("-", "_")
    aliases = {
        "logp": "mollogp",
        "mw": "molwt",
        "hac": "heavyatomcount",
        "heavy_atom_count": "heavyatomcount",
        "bertz": "bertzct",
        "nrot": "numrotatablebonds",
        "rotbonds": "numrotatablebonds",
        "logp_mw": "mollogp_molwt",
        "mollogp+molwt": "mollogp_molwt",
    }
    key = aliases.get(key, key)
    if key not in PROXY_NAMES:
        raise ValueError(f"Unknown MGW_PROXY={name!r}; choose one of {PROXY_NAMES}")
    return key


def mw_oversize_penalty(
    mw: float,
    *,
    start: float = DEFAULT_MW_PENALTY_START,
    tau: float = DEFAULT_MW_PENALTY_TAU,
) -> float:
    """Exponential soft penalty for MW above ``start``: ``expm1((mw-start)/tau)``.

    Returns 0 when ``mw <= start`` or when ``start <= 0`` (disabled).
    """
    if float(start) <= 0.0:
        return 0.0
    excess = float(mw) - float(start)
    if excess <= 0.0:
        return 0.0
    t = max(float(tau), 1e-6)
    return float(np.expm1(excess / t))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def proxy_from_mol(mol, proxy: str) -> float:
    """Raw unbounded proxy value (higher = better for RL)."""
    if proxy == "mollogp":
        return float(Descriptors.MolLogP(mol))
    if proxy == "molwt":
        return float(Descriptors.MolWt(mol))
    if proxy == "heavyatomcount":
        return float(mol.GetNumHeavyAtoms())
    if proxy == "bertzct":
        return float(Descriptors.BertzCT(mol))
    if proxy == "numrotatablebonds":
        return float(Descriptors.NumRotatableBonds(mol))
    if proxy == "tpsa":
        return float(Descriptors.TPSA(mol))
    if proxy == "mollogp_molwt":
        return float(Descriptors.MolLogP(mol)) + float(Descriptors.MolWt(mol))
    raise ValueError(proxy)


class UnboundedProxyScorer:
    def __init__(
        self,
        proxy: str | None = None,
        *,
        mw_penalty_start: float | None = None,
        mw_penalty_tau: float | None = None,
    ):
        self.proxy = normalize_proxy_name(
            proxy if proxy is not None else os.environ.get("MGW_PROXY", "mollogp")
        )
        self.mw_penalty_start = (
            float(mw_penalty_start)
            if mw_penalty_start is not None
            else _env_float("MGW_MW_PENALTY_START", DEFAULT_MW_PENALTY_START)
        )
        self.mw_penalty_tau = (
            float(mw_penalty_tau)
            if mw_penalty_tau is not None
            else _env_float("MGW_MW_PENALTY_TAU", DEFAULT_MW_PENALTY_TAU)
        )

    def compute_batch(self, smiles_list: list[str]) -> UnboundedProxyBatch:
        n = len(smiles_list)
        if n == 0:
            return UnboundedProxyBatch([], [], [], [], [], [], [], [])

        score_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        logp_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        mw_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        hac_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        bertz_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        gate_ok = [False] * n
        gated_idx: list[int] = []

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not atom_check(smi):
                continue
            try:
                lp = float(Descriptors.MolLogP(mol))
                mw = float(Descriptors.MolWt(mol))
                hac = float(mol.GetNumHeavyAtoms())
                bz = float(Descriptors.BertzCT(mol))
                val = proxy_from_mol(mol, self.proxy)
                pen = mw_oversize_penalty(
                    mw,
                    start=self.mw_penalty_start,
                    tau=self.mw_penalty_tau,
                )
                score = val - pen
            except Exception:
                continue
            gate_ok[i] = True
            logp_arr[i] = lp
            mw_arr[i] = mw
            hac_arr[i] = hac
            bertz_arr[i] = bz
            score_arr[i] = score
            gated_idx.append(i)

        policy = np.full(n, INVALID_PENALTY, dtype=np.float64)
        if gated_idx:
            z = _zscore(score_arr[gated_idx])
            for j, i in enumerate(gated_idx):
                policy[i] = float(z[j])

        return UnboundedProxyBatch(
            policy=[float(x) for x in policy.tolist()],
            raw_total=[float(x) for x in score_arr.tolist()],
            quality=[float(x) for x in score_arr.tolist()],
            gate_ok=gate_ok,
            mollogp=[float(x) for x in logp_arr.tolist()],
            molwt=[float(x) for x in mw_arr.tolist()],
            heavyatomcount=[float(x) for x in hac_arr.tolist()],
            bertzct=[float(x) for x in bertz_arr.tolist()],
        )


def get_unbounded_proxy_scorer() -> UnboundedProxyScorer:
    # Re-read env each call so multi-proxy sweeps don't need process restart tricks.
    return UnboundedProxyScorer()
