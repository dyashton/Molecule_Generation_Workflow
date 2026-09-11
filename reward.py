"""
NLRP objective suite for REINFORCE fine-tuning.

Composes migrated objectives from `objectives/`.

- `compute_rewards`: inference-friendly API returning **z-scored** per-SMILES rewards
  (used by `generate.py` and as the policy signal in `train.py`).
- `compute_reward_batch`: training helper that returns **both** z-scored policy rewards
  and **raw** (unnormalized) objective components for logging / checkpoint selection.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from objectives.binding_affinity import BindingAffinityScorer, predict_sync
from objectives.co2_reward import CO2ObjectiveScorer
from objectives.diversity import tanimoto_diversity_scores
from objectives.druglike_reward import get_druglike_scorer
from objectives.unbounded_proxy_reward import get_unbounded_proxy_scorer
from objectives.filters import FilterConfig, passes_filters
from objectives.shape_exclusion import ShapeExclusionScreener
from objectives.synthesizability import SynthesizabilityScorer
from dotenv import load_dotenv

load_dotenv()

INVALID_PENALTY = -1.0

# Same weights as `Final_Molecule_Gen/rl.py`
BA_WEIGHT = 0.7
DIVERSITY_WEIGHT = 0.3


_BA_SCORER: BindingAffinityScorer | None = None
_SE_SCREENER: ShapeExclusionScreener | None = None
_SA_SCORER: SynthesizabilityScorer | None = None
_CO2_SCORER: CO2ObjectiveScorer | None = None
_CO2_SCORER_KWARGS: dict = {
    "random_seed": 42,
    "n_conformers": 2,
    "scaffold_bucket": 0,
    "scaffold_penalty": 0.0,
    "use_scaffold_filter": False,
    "n_workers": 0,
    "include_pfc": True,
    "use_objective_ceilings": False,
    "use_sigmoid_reward": False,
    "prior_fp_floor": 0.0,
}


@dataclass(frozen=True)
class RewardBatch:
    """Per-SMILES outputs aligned with the input `smiles_list` order."""

    policy: list[float]  # z-scored BA/div mix (REINFORCE signal)
    raw_total: list[float]  # BA_WEIGHT * raw_ba + DIVERSITY_WEIGHT * raw_div (unnormalized)
    raw_ba: list[float]
    raw_div: list[float]
    gate_ok: list[bool]
    # CO2: physchem + PFC (no diversity). Druglike: −SCScore. NLRP: unused (nans).
    raw_physchem: list[float]
    # CO2: pre-DF quality for elite replay. Druglike: raw_total when gated. NLRP: nans.
    quality: list[float]
    # CO2: multiplicative MW scale [0,1]. Druglike: MolLogP. NLRP: nans.
    mw_scale: list[float]


def _profile_rewards_enabled() -> bool:
    v = os.environ.get("MGW_REWARD_PROFILE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _get_scorers():
    global _BA_SCORER, _SE_SCREENER, _SA_SCORER
    if _BA_SCORER is None:
        _BA_SCORER = BindingAffinityScorer()
    if _SE_SCREENER is None:
        _SE_SCREENER = ShapeExclusionScreener()
    if _SA_SCORER is None:
        _SA_SCORER = SynthesizabilityScorer()
    return _BA_SCORER, _SE_SCREENER, _SA_SCORER


def _objective_name() -> str:
    """
    Select which objective suite to use.

    - MGW_OBJECTIVE=nlrp (default)
    - MGW_OBJECTIVE=co2
    - MGW_OBJECTIVE=druglike
    - MGW_OBJECTIVE=unbounded_proxy  (needs MGW_PROXY=...)
    """
    return os.environ.get("MGW_OBJECTIVE", "nlrp").strip().lower()


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def configure_co2_scorer(**kwargs) -> None:
    """Reset CO2 scorer with updated kwargs (conformers, workers, …)."""
    global _CO2_SCORER, _CO2_SCORER_KWARGS
    if _CO2_SCORER is not None:
        _CO2_SCORER.close()
    # Env defaults unless caller passes flags explicitly.
    merged = {**_CO2_SCORER_KWARGS, **kwargs}
    if "include_pfc" not in kwargs:
        merged["include_pfc"] = _env_flag("MGW_INCLUDE_PFC", default=True)
    if "use_objective_ceilings" not in kwargs:
        merged["use_objective_ceilings"] = _env_flag(
            "MGW_OBJECTIVE_CEILINGS", default=False
        )
    if "use_sigmoid_reward" not in kwargs:
        merged["use_sigmoid_reward"] = _env_flag("MGW_CO2_SIGMOID", default=False)
    if "apply_additive_terms" not in kwargs:
        merged["apply_additive_terms"] = not _env_flag(
            "MGW_NO_PENALTIES", default=False
        )
    _CO2_SCORER_KWARGS = merged
    _CO2_SCORER = None


def ensure_co2_scorer() -> CO2ObjectiveScorer:
    """
    Build the CO2 scorer (and process pool) immediately.

    Call this before CUDA init when REWARD_WORKERS > 0 so the pool can fork safely.
    """
    return _get_co2_scorer()


def _get_co2_scorer() -> CO2ObjectiveScorer:
    global _CO2_SCORER
    if _CO2_SCORER is None:
        _CO2_SCORER = CO2ObjectiveScorer(**_CO2_SCORER_KWARGS)
    return _CO2_SCORER


def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + 1e-8)


def compute_reward_batch(smiles_list: list[str]) -> RewardBatch:
    """
    Compute rewards for a batch.

    Policy rewards match the ported logic in `Final_Molecule_Gen/rl.py`:
      - z-score `raw_ba` and `raw_div` within the batch, then combine with 0.7/0.3.

    Raw totals are the **unnormalized** mix:
      `BA_WEIGHT * raw_ba + DIVERSITY_WEIGHT * raw_div`
    """
    if _objective_name() == "co2":
        batch = _get_co2_scorer().compute_batch(smiles_list)
        return RewardBatch(
            policy=batch.policy,
            raw_total=batch.raw_total,
            raw_ba=[float("nan")] * len(smiles_list),
            raw_div=[float("nan")] * len(smiles_list),
            gate_ok=batch.gate_ok,
            raw_physchem=batch.physchem,
            quality=batch.quality,
            mw_scale=batch.mw_scale,
        )

    if _objective_name() == "druglike":
        # Slot map: raw_ba=logit(QED), raw_div=−log freq, raw_physchem=−SCScore, mw_scale=MolLogP.
        batch = get_druglike_scorer().compute_batch(smiles_list)
        return RewardBatch(
            policy=batch.policy,
            raw_total=batch.raw_total,
            raw_ba=batch.qed,
            raw_div=batch.diversity,
            gate_ok=batch.gate_ok,
            raw_physchem=batch.sa_good,
            quality=batch.quality,
            mw_scale=batch.logp_soft,
        )

    if _objective_name() == "unbounded_proxy":
        # Slot map: raw_total=proxy; companions raw_ba=MolLogP, raw_physchem=MolWt,
        # mw_scale=HeavyAtomCount, raw_div=BertzCT.
        batch = get_unbounded_proxy_scorer().compute_batch(smiles_list)
        return RewardBatch(
            policy=batch.policy,
            raw_total=batch.raw_total,
            raw_ba=batch.mollogp,
            raw_div=batch.bertzct,
            gate_ok=batch.gate_ok,
            raw_physchem=batch.molwt,
            quality=batch.quality,
            mw_scale=batch.heavyatomcount,
        )

    profile = _profile_rewards_enabled()
    t0 = time.perf_counter() if profile else 0.0

    ba_scorer, se_screener, sa_scorer = _get_scorers()
    t_after_scorers = time.perf_counter() if profile else 0.0

    n = len(smiles_list)
    if n == 0:
        return RewardBatch(
            policy=[],
            raw_total=[],
            raw_ba=[],
            raw_div=[],
            gate_ok=[],
            raw_physchem=[],
            quality=[],
            mw_scale=[],
        )

    t_parse0 = time.perf_counter() if profile else 0.0
    mols: list[Chem.Mol | None] = [Chem.MolFromSmiles(s) for s in smiles_list]
    t_parse1 = time.perf_counter() if profile else 0.0

    t_div0 = time.perf_counter() if profile else 0.0
    diversity = tanimoto_diversity_scores(smiles_list)
    t_div1 = time.perf_counter() if profile else 0.0

    sa_scores: list[float | None] = [None] * n
    shape_fits: list[bool] = [False] * n
    gate_ok: list[bool] = [False] * n

    cfg = FilterConfig()
    t_sa = 0.0
    t_shape = 0.0
    t_filter = 0.0
    n_valid = 0
    n_shape_ok = 0
    n_gated = 0
    for i, (smi, mol) in enumerate(zip(smiles_list, mols)):
        if mol is None:
            continue
        n_valid += 1
        if profile:
            t_a = time.perf_counter()
        sa = sa_scorer.score(smi)
        if profile:
            t_sa += time.perf_counter() - t_a
        sa_scores[i] = sa
        if profile:
            t_b = time.perf_counter()
        fit = se_screener.single_smiles_shape_exclusion_check(smi)
        if profile:
            t_shape += time.perf_counter() - t_b
        shape_fits[i] = bool(fit)
        if fit:
            n_shape_ok += 1
        if profile:
            t_c = time.perf_counter()
        ok = bool(fit) and passes_filters(smi, mol=mol, sa_score=sa, cfg=cfg)
        if profile:
            t_filter += time.perf_counter() - t_c
        gate_ok[i] = ok
        if ok:
            n_gated += 1

    ba_raw = np.full((n,), INVALID_PENALTY, dtype=np.float64)
    idx_to_score = [i for i in range(n) if gate_ok[i]]
    t_ba = 0.0
    if idx_to_score:
        gated_smiles = [smiles_list[i] for i in idx_to_score]
        if profile:
            t_ba0 = time.perf_counter()
        preds = predict_sync(ba_scorer, gated_smiles)
        if profile:
            t_ba = time.perf_counter() - t_ba0
        for i, p in zip(idx_to_score, preds):
            ba_raw[i] = float(p)

    t_post0 = time.perf_counter() if profile else 0.0
    div_raw = diversity.astype(np.float64)
    for i, mol in enumerate(mols):
        if mol is None:
            div_raw[i] = INVALID_PENALTY

    raw_total = (BA_WEIGHT * ba_raw + DIVERSITY_WEIGHT * div_raw).astype(np.float64)

    ba_norm = _zscore(ba_raw)
    div_norm = _zscore(div_raw)
    policy = (BA_WEIGHT * ba_norm + DIVERSITY_WEIGHT * div_norm).astype(np.float64)
    t_post1 = time.perf_counter() if profile else 0.0

    if profile:
        t_end = time.perf_counter()
        print(
            "[MGW_REWARD_PROFILE] "
            f"n={n} valid={n_valid} shape_ok={n_shape_ok} gated={n_gated} "
            f"scorers_init_s={t_after_scorers - t0:.3f} "
            f"parse_s={t_parse1 - t_parse0:.3f} "
            f"diversity_s={t_div1 - t_div0:.3f} "
            f"sa_s={t_sa:.3f} shape_s={t_shape:.3f} filter_s={t_filter:.3f} "
            f"ba_s={t_ba:.3f} "
            f"post_s={t_post1 - t_post0:.3f} "
            f"total_s={t_end - t0:.3f}",
            flush=True,
        )

    return RewardBatch(
        policy=[float(x) for x in policy.tolist()],
        raw_total=[float(x) for x in raw_total.tolist()],
        raw_ba=[float(x) for x in ba_raw.tolist()],
        raw_div=[float(x) for x in div_raw.tolist()],
        gate_ok=list(gate_ok),
        raw_physchem=[float("nan")] * n,
        quality=[float("nan")] * n,
        mw_scale=[float("nan")] * n,
    )


def compute_rewards(smiles_list: list[str]) -> list[float]:
    """Return z-scored policy rewards (same length as `smiles_list`)."""
    return compute_reward_batch(smiles_list).policy
