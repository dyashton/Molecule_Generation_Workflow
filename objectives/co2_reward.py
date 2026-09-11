from __future__ import annotations

"""
CO2 objective: soft-bounded z-scored physchem (+ optional PFC) + diversity, with
optional Morgan-FP soft floor vs a cached prior SMILES bank (anti-OOD tether).
Scaffold DF is unused unless use_scaffold_filter and bucket > 0.

Additive soft penalties (after physchem+div): MW trapezoid (flat 100–350,
outer 50–400), SCScore (full ≤3.5, unit excess at 5), max consecutive rotatable
bonds (full ≤4, unit at 7), total rotatable-bond window (ideal 3–6; excess
outside), amine-justified OH (excess = max(0, n_OH − n_amine −
FREE), FREE=0, uncapped, weight 2.0), soft amine floor (excess = max(0, 3 − n_amine) / 3),
min non-bonded N···N distance (excess when d < 2.8 Å, graph dist ≥ 4),
soft SMARTS alerts
(epoxide/aziridine, sulfur, esters, carboxylic acids, cyclic acetals, aminals,
hydrates, hemiaminals, peroxides, peracids, amides, hindered amine α-carbons,
ring-oxygen count, alkynes (weight 2.0), aromatic furans — no PAINS catalog), soft non-basic N
count (excess = max(0, n_N − n_basic_amine)).
Soft bonuses: azacrown (min(2, max(0, max-ring-aliphatic-N − 2))) and
vicinal primary diamine-diol (binary, weight 2.0).
Each axis contributes weight × excess with default weight 1.0 (MW weight 5.0,
OH weight 2.0, alkyne weight 2.0).
PFC weight is 1.25×.

Extensive Mordred terms (apol, TopoPSA, basicity) are scored per heavy atom
(raw / n_HA) before z/sigmoid so physchem tracks polar/basic *density*, not bulk.
LogP, HallKierAlpha, and PFC are already intensive and are unchanged.
The old HAE multiplier is not applied (would double-count /n_HA).

Set include_pfc=False (or MGW_INCLUDE_PFC=0) to skip Polar Focal Concavity entirely.
Set use_objective_ceilings=True (or MGW_OBJECTIVE_CEILINGS=1) to clamp raw properties
to OBJECTIVE_CEILINGS before z-scoring.
Set use_sigmoid_reward=True (or MGW_CO2_SIGMOID=1) for weighted sum of
sigmoid_to_unit(raw, *RANGES[key]) terms (legacy CO2 scheme; LogP inverted).
Set apply_additive_terms=False (or MGW_NO_PENALTIES=1) to skip additive
penalties other than molecular weight, plus bonuses; reward is then
physchem + diversity − MW penalty (hard gate and prior-FP floor still apply).

Z-score refs: mean/std on GuacaMol-prior gated samples
(publication/co2/data/scored_components_nconf2.csv, n≈8092).
Each z-term is tanh-soft-capped at Z_SOFT to limit outlier inflation.

MW / SA / consec-rot / total-rot-window / amine-justified-OH / amine-floor /
NN-distance / SMARTS alerts / non-basic N are additive ramping penalties
(excess keeps growing). Hard gate is charge 0, atoms in {C,H,N,O,F,S},
no sulfonyl, one fragment.
Prior-FP tether remains multiplicative. Scaffold DF is unused (bucket 0).

Per-molecule work (gate / Mordred / PFC) can run in a ProcessPool; diversity,
prior-FP tether, SCScore, consec-rot, and OH/amine counts stay on
the main process.
"""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from mordred import Calculator
from mordred.EState import AtomTypeEState
from mordred.Polarizability import APol
from mordred.TopoPSA import TopoPSA
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

from objectives.co2_polar_focal_concavity import PolarFocalConcavity
from objectives.co2_rl_helpers import (
    ScaffoldDiversityFilter,
    amine_floor_excess,
    amine_justified_oh_excess,
    azacrown_bonus_units,
    co2_gate_ok,
    count_amine_n,
    count_oh,
    count_rotatable_bonds,
    high_end_excess,
    per_heavy_atom,
    max_consecutive_rotatable_bonds,
    min_nonbonded_nn_distance_over_confs,
    mw_excess,
    mw_scale,
    nn_distance_excess,
    nonbasic_n_excess,
    rot_window_excess,
    substructure_alert_excesses,
    vicinal_diamine_diol,
)
from objectives.prior_fp_tether import PriorFPTether
from objectives.synthesizability import SynthesizabilityScorer


# Soft limits / sigmoid anchors: union of GuacaMol-prior and AxelRolov solvent
# p10/p90 boxes — lo=min, hi=max per term. REF_STATS remain GuacaMol-prior gated mean/std.
# AxelRolov scored: publication/co2/data/axelrolov_scored_nconf2.csv (n=945).
# RANGE_PAD extends each side by pad * (hi - lo); diversity stays [0, 1].
RANGE_PAD = 0.05


def pad_range(
    lo: float, hi: float, *, pad: float = RANGE_PAD
) -> tuple[float, float]:
    """Widen [lo, hi] by `pad` × span on each side."""
    span = float(hi) - float(lo)
    return (float(lo) - float(pad) * span, float(hi) + float(pad) * span)


_BASE_RANGES: dict[str, tuple[float, float]] = {
    # apol / TopoPSA / basicity are per-heavy-atom (raw / n_HA).
    "apol": (1.894, 2.672),
    "TopoPSA": (1.52, 4.211),
    "basicity": (0.0, 1.515),
    "LogP": (-0.63, 5.29),
    "HallKierAlpha": (-3.61, -0.08),
    "PFC_composite": (0.35, 0.74),
    "diversity": (0.0, 1.0),
    # Unused for scoring (legacy HAE multiplier); kept for diagnostics.
    "HAE": (-0.421, 0.0913),
}

RANGES: dict[str, tuple[float, float]] = {
    key: (lo, hi) if key == "diversity" else pad_range(lo, hi)
    for key, (lo, hi) in _BASE_RANGES.items()
}

# When use_objective_ceilings=True, clamp raw values to these [lo, hi] before z-scoring.
OBJECTIVE_CEILINGS: dict[str, tuple[float, float]] = dict(RANGES)

SIGMOID_K = 16.0

# Pretrained-gated mean/std (population std, ddof=0).
REF_STATS: dict[str, tuple[float, float]] = {
    # apol / TopoPSA / basicity: GuacaMol-prior gated mean/std of col/n_HA (ddof=0).
    "apol": (2.09400, 0.175382),
    "TopoPSA": (2.83357, 1.12280),
    "basicity": (0.390902, 0.344405),
    "LogP": (3.28635, 1.59979),
    "HallKierAlpha": (-2.47535, 0.901032),
    "PFC_composite": (0.597969, 0.132485),
    "diversity": (0.883852, 0.0185346),
    # Unused for scoring (legacy HAE multiplier).
    "HAE": (0.0504954, 0.113775),
}

# Equal weight on physchem/diversity z-terms; PFC mildly emphasized.
W_TERM = 1.0
W_DIVERSE = W_TERM
W_APOL = W_TERM
W_TPSA = W_TERM
W_BASICITY = W_TERM
W_LOGP = W_TERM
W_HALL_KIER_ALPHA = W_TERM
W_PFC_COMPOSITE = 1.25 * W_TERM

MW_SCALE_LO = 50.0
MW_SCALE_HI = 400.0
MW_FLAT_LO = 100.0
MW_FLAT_HI = 350.0

SA_SCALE_FULL_HI = 3.5
SA_SCALE_ZERO_HI = 5.0
CONSEC_ROT_FULL_HI = 4.0
CONSEC_ROT_ZERO_HI = 7.0
ROT_WINDOW_LO = 3.0  # total Lipinski rotatable bonds ideal window
ROT_WINDOW_HI = 6.0
OH_SCALE_FULL_HI = 3.0  # legacy saturating scale (diagnostics / figures)
OH_SCALE_ZERO_HI = 5.0
OH_AMINE_FREE = 0.0  # amine-justified budget = n_amine + FREE
AMINE_FLOOR = 3.0  # soft floor: excess = max(0, floor - n_amine) / floor

# Additive penalty weights: subtract weight × excess per axis.
W_MW_PENALTY = 5.0
W_SA_PENALTY = 1.0
W_CONSEC_ROT_PENALTY = 1.0
W_ROT_WINDOW_PENALTY = 1.0
W_OH_PENALTY = 2.0
W_NN_PENALTY = 1.0
W_AMINE_FLOOR_PENALTY = 1.0
W_EPOXIDE_PENALTY = 1.0
W_SULFUR_PENALTY = 1.0
W_ESTER_PENALTY = 1.0
W_ACETAL_PENALTY = 1.0
W_AMINAL_PENALTY = 1.0
W_HYDRATE_PENALTY = 1.0
W_HEMIAMINAL_PENALTY = 1.0
W_NONBASIC_N_PENALTY = 1.0
W_PEROXIDE_PENALTY = 1.0
W_PERACID_PENALTY = 1.0
W_CARBOXYLIC_ACID_PENALTY = 1.0
W_AMIDE_PENALTY = 1.0
W_HINDERED_AMINE_PENALTY = 1.0
W_RING_O_PENALTY = 1.0
W_ALKYNE_PENALTY = 2.0
W_FURAN_PENALTY = 1.0
W_AZACROWN_BONUS = 1.0
W_VICINAL_DIOL_BONUS = 2.0
ACETAL_FREE = 1.0  # cyclic acetals allowed before excess
SULFUR_FREE = 0.0  # no free sulfur
RING_O_FREE = 1.0  # one ring oxygen allowed before excess

# Legacy HAE multiplier knobs (unused in scoring; helpers still tested).
W_HAE_ALPHA = 0.35
HAE_MULT_LO = 0.25
HAE_MULT_HI = 3.0

NN_DIST_THRESH = 2.8

# Process-pool worker state (set by initializer).
_WORKER_CALC: Calculator | None = None
_WORKER_PFC: PolarFocalConcavity | None = None
_WORKER_INCLUDE_PFC: bool = True
_WORKER_USE_OBJECTIVE_CEILINGS: bool = False
_WORKER_USE_SIGMOID_REWARD: bool = False


@dataclass(frozen=True)
class CO2RewardBatch:
    policy: list[float]
    raw_total: list[float]
    gate_ok: list[bool]
    # Physchem + PFC only (excludes diversity). Aligned with policy/raw_total.
    physchem: list[float]
    # Pre-scaffold-filter full reward (for elite replay; post prior-FP floor). NaN if not gated.
    quality: list[float]
    # Multiplicative MW scale in [0, 1]. NaN if not gated.
    mw_scale: list[float]


@dataclass(frozen=True)
class CO2LocalScore:
    """Per-molecule components with no diversity / scaffold side effects."""

    gate_ok: bool
    physchem_z: float
    mw_scale: float
    can: str | None
    hae_raw: float = float("nan")
    hae_mult: float = float("nan")
    n_ha: int = 0
    n_amine: int = 0
    min_nn_distance: float = float("nan")
    # Stage timings (seconds); only populated when profiling.
    t_gate: float = 0.0
    t_mordred: float = 0.0
    t_pfc: float = 0.0


def _embed_min_nn_distance(smi: str, *, random_seed: int = 42) -> float:
    """Cheap single-conformer N···N check when PFC is off. NaN on embed failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float("nan")
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.numThreads = 1
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        return float("nan")
    return float(min_nonbonded_nn_distance_over_confs(mol_h))


def sigmoid_to_unit(raw: float, lo: float, hi: float, k: float = SIGMOID_K) -> float:
    span = max(hi - lo, 1e-9)
    t = (raw - lo) / span
    return float(1.0 / (1.0 + math.exp(-k * (t - 0.5))))


# Soft-bound raw z so outlier chase cannot inflate rewards to ±10σ+.
Z_SOFT = 3.0


def zscore(raw: float, mu: float, sigma: float) -> float:
    """Standardized score soft-capped with tanh (range ≈ (-Z_SOFT, Z_SOFT))."""
    z = (float(raw) - float(mu)) / max(float(sigma), 1e-8)
    return float(Z_SOFT * math.tanh(z / Z_SOFT))


def zscore_key(raw: float, key: str, *, use_ceilings: bool = False) -> float:
    x = float(raw)
    if use_ceilings and key in OBJECTIVE_CEILINGS:
        lo, hi = OBJECTIVE_CEILINGS[key]
        x = min(float(hi), max(float(lo), x))
    mu, sigma = REF_STATS[key]
    return zscore(x, mu, sigma)


def sigmoid_key(raw: float, key: str) -> float:
    """Map raw property to (0, 1) via RANGES anchors (legacy CO2 scheme)."""
    lo, hi = RANGES[key]
    return sigmoid_to_unit(float(raw), float(lo), float(hi))


def term_score(
    raw: float,
    key: str,
    *,
    use_sigmoid: bool = False,
    use_ceilings: bool = False,
) -> float:
    """Unit contribution before weight / LogP handling."""
    if use_sigmoid:
        return sigmoid_key(raw, key)
    return zscore_key(raw, key, use_ceilings=use_ceilings)


def _physchem_weight_sum(*, include_pfc: bool) -> float:
    w = (
        W_APOL
        + W_TPSA
        + W_BASICITY
        + W_LOGP
        + W_HALL_KIER_ALPHA
    )
    if include_pfc:
        w += W_PFC_COMPOSITE
    return float(w)


def _safe(x):
    return float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else 0.0


def _profile_rewards_enabled() -> bool:
    v = os.environ.get("MGW_REWARD_PROFILE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _make_mordred_calculator() -> Calculator:
    return Calculator(
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


def score_co2_local(
    smi: str,
    calc: Calculator,
    pfc: PolarFocalConcavity | None,
    *,
    include_pfc: bool = True,
    use_objective_ceilings: bool = False,
    use_sigmoid_reward: bool = False,
    profile: bool = False,
) -> CO2LocalScore:
    """Gate + Mordred (+ optional PFC) for one SMILES. No diversity / scaffold state."""
    t_gate = t_mordred = t_pfc = 0.0
    t0 = time.perf_counter() if profile else 0.0

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return CO2LocalScore(
            gate_ok=False,
            physchem_z=float("nan"),
            mw_scale=float("nan"),
            can=None,
            t_gate=(time.perf_counter() - t0) if profile else 0.0,
        )

    n_ha = int(mol.GetNumHeavyAtoms())
    n_amine = int(count_amine_n(mol))
    ok = co2_gate_ok(mol)
    if profile:
        t_gate = time.perf_counter() - t0
    if not ok:
        return CO2LocalScore(
            gate_ok=False,
            physchem_z=float("nan"),
            mw_scale=float("nan"),
            can=None,
            n_ha=n_ha,
            n_amine=n_amine,
            t_gate=t_gate,
        )

    mw = float(Descriptors.MolWt(mol))
    scale = mw_scale(
        mw,
        lo=MW_SCALE_LO,
        hi=MW_SCALE_HI,
        inner_lo=MW_FLAT_LO,
        inner_hi=MW_FLAT_HI,
    )

    t_m0 = time.perf_counter() if profile else 0.0
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
    if profile:
        t_mordred = time.perf_counter() - t_m0

    ceil = bool(use_objective_ceilings) and not bool(use_sigmoid_reward)
    sig = bool(use_sigmoid_reward)
    pfc_unit = 0.0
    min_nn = float("nan")
    if include_pfc:
        if pfc is None:
            raise ValueError("include_pfc=True requires a PolarFocalConcavity instance")
        t_p0 = time.perf_counter() if profile else 0.0
        try:
            pfc_out = pfc.compute(smi)
            pfc_raw = float(pfc_out["composite"])
            min_nn = float(pfc_out.get("min_nn_distance", float("nan")))
        except Exception:
            pfc_raw = float(REF_STATS["PFC_composite"][0])
            min_nn = float("nan")
        if profile:
            t_pfc = time.perf_counter() - t_p0
        pfc_unit = term_score(
            pfc_raw, "PFC_composite", use_sigmoid=sig, use_ceilings=ceil
        )
    else:
        # No PFC embed — one cheap conformer for N···N soft penalty only.
        min_nn = _embed_min_nn_distance(smi)

    apol_i = per_heavy_atom(apol, n_ha)
    tpsa_i = per_heavy_atom(tpsa, n_ha)
    bas_i = per_heavy_atom(basicity, n_ha)
    bas_u = term_score(bas_i, "basicity", use_sigmoid=sig, use_ceilings=ceil)
    tpsa_u = term_score(tpsa_i, "TopoPSA", use_sigmoid=sig, use_ceilings=ceil)
    # Diagnostic: sum of intensive unit scores (not /n_HA again).
    hae = float(bas_u) + float(tpsa_u)
    if include_pfc:
        hae += float(pfc_unit)
    m = 1.0

    if sig:
        # All unit terms in (0, 1); LogP prefers low → (1 - sigmoid). Weighted mean ∈ (0, 1).
        numer = (
            W_APOL * term_score(apol_i, "apol", use_sigmoid=True)
            + W_TPSA * tpsa_u
            + W_BASICITY * bas_u
            + W_LOGP * (1.0 - term_score(logp, "LogP", use_sigmoid=True))
            + W_HALL_KIER_ALPHA * term_score(alpha, "HallKierAlpha", use_sigmoid=True)
        )
        if include_pfc:
            numer += W_PFC_COMPOSITE * pfc_unit
        physchem_z = float(numer / _physchem_weight_sum(include_pfc=include_pfc))
    else:
        physchem_z = (
            W_APOL * term_score(apol_i, "apol", use_ceilings=ceil)
            + W_TPSA * tpsa_u
            + W_BASICITY * bas_u
            + W_LOGP * term_score(logp, "LogP", use_ceilings=ceil) * -1.0
            + W_HALL_KIER_ALPHA
            * term_score(alpha, "HallKierAlpha", use_ceilings=ceil)
            + (W_PFC_COMPOSITE * pfc_unit if include_pfc else 0.0)
        )
    return CO2LocalScore(
        gate_ok=True,
        physchem_z=float(physchem_z),
        mw_scale=float(scale),
        can=Chem.MolToSmiles(mol),
        hae_raw=float(hae),
        hae_mult=float(m),
        n_ha=n_ha,
        n_amine=n_amine,
        min_nn_distance=float(min_nn),
        t_gate=t_gate,
        t_mordred=t_mordred,
        t_pfc=t_pfc,
    )



def _mp_context():
    """Prefer fork (low IPC). train.py must build the pool before CUDA init."""
    try:
        return get_context("fork")
    except ValueError:
        return get_context("spawn")


def _init_reward_worker(
    random_seed: int,
    n_conformers: int,
    embed_num_threads: int,
    include_pfc: bool = True,
    use_objective_ceilings: bool = False,
    use_sigmoid_reward: bool = False,
) -> None:
    """ProcessPool initializer: one Mordred(+optional PFC) scorer per worker."""
    global _WORKER_CALC, _WORKER_PFC, _WORKER_INCLUDE_PFC
    global _WORKER_USE_OBJECTIVE_CEILINGS, _WORKER_USE_SIGMOID_REWARD
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _WORKER_CALC = _make_mordred_calculator()
    _WORKER_INCLUDE_PFC = bool(include_pfc)
    _WORKER_USE_OBJECTIVE_CEILINGS = bool(use_objective_ceilings)
    _WORKER_USE_SIGMOID_REWARD = bool(use_sigmoid_reward)
    if _WORKER_INCLUDE_PFC:
        _WORKER_PFC = PolarFocalConcavity(
            conformer_seed=int(random_seed),
            n_conformers=int(n_conformers),
            embed_num_threads=int(embed_num_threads),
        )
    else:
        _WORKER_PFC = None


def _worker_score_co2_local(smi: str) -> CO2LocalScore:
    assert _WORKER_CALC is not None
    if _WORKER_INCLUDE_PFC:
        assert _WORKER_PFC is not None
    return score_co2_local(
        smi,
        _WORKER_CALC,
        _WORKER_PFC,
        include_pfc=_WORKER_INCLUDE_PFC,
        use_objective_ceilings=_WORKER_USE_OBJECTIVE_CEILINGS,
        use_sigmoid_reward=_WORKER_USE_SIGMOID_REWARD,
        profile=False,
    )


def tanimoto_diversity_scores(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    n = len(smiles_list)
    fps: list = [None] * n
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps[i] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    valid_idx = [i for i in range(n) if fps[i] is not None]
    scores = np.zeros(n, dtype=np.float64)
    if not valid_idx:
        return scores
    # ponytail: scale by canonical uniqueness; singleton gets 0 (was 1.0 → collapse reward)
    canonical = []
    for i in valid_idx:
        mol = Chem.MolFromSmiles(smiles_list[i])
        canonical.append(Chem.MolToSmiles(mol) if mol is not None else smiles_list[i])
    uniq_scale = len(set(canonical)) / float(len(valid_idx))
    if len(valid_idx) == 1:
        scores[valid_idx[0]] = 0.0
        return scores
    for i in valid_idx:
        sims = [
            DataStructs.TanimotoSimilarity(fps[i], fps[j])
            for j in valid_idx
            if j != i
        ]
        scores[i] = float(np.mean([1.0 - s for s in sims])) * uniq_scale
    return scores


class CO2ObjectiveScorer:
    def __init__(
        self,
        *,
        random_seed: int = 42,
        n_conformers: int = 2,
        scaffold_bucket: int = 0,
        scaffold_penalty: float = 0.0,
        scaffold_min_score: float = float("-inf"),
        use_scaffold_filter: bool = False,
        n_workers: int = 0,
        include_pfc: bool = True,
        use_objective_ceilings: bool = False,
        use_sigmoid_reward: bool = False,
        apply_additive_terms: bool = True,
        prior_fp_floor: float = 0.0,
    ):
        self.random_seed = int(random_seed)
        self.n_conformers = int(n_conformers)
        self.n_workers = max(int(n_workers), 0)
        self.include_pfc = bool(include_pfc)
        self.use_objective_ceilings = bool(use_objective_ceilings)
        self.use_sigmoid_reward = bool(use_sigmoid_reward)
        self.apply_additive_terms = bool(apply_additive_terms)
        self.calc = _make_mordred_calculator()
        # Serial path: RDKit may use all cores per embed (embed_num_threads=0).
        self.pfc: PolarFocalConcavity | None = None
        if self.include_pfc:
            self.pfc = PolarFocalConcavity(
                conformer_seed=self.random_seed, n_conformers=self.n_conformers
            )
        self.scaffold_filter: ScaffoldDiversityFilter | None = None
        if use_scaffold_filter and int(scaffold_bucket) > 0:
            self.scaffold_filter = ScaffoldDiversityFilter(
                bucket_size=int(scaffold_bucket),
                penalty=float(scaffold_penalty),
                min_score=float(scaffold_min_score),
            )
        # Bank FPs attached later via set_prior_fp_bank (cache load / prior sample).
        self.prior_fp_tether = PriorFPTether(floor=float(prior_fp_floor))
        self._sa_scorer: SynthesizabilityScorer | None = None
        self._pool: ProcessPoolExecutor | None = None
        if self.n_workers > 0:
            # fork: build this pool before CUDA init (see train.py). spawn is a fallback.
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=_mp_context(),
                initializer=_init_reward_worker,
                initargs=(
                    self.random_seed,
                    self.n_conformers,
                    1,
                    self.include_pfc,
                    self.use_objective_ceilings,
                    self.use_sigmoid_reward,
                ),
            )

    def _get_sa_scorer(self) -> SynthesizabilityScorer:
        if self._sa_scorer is None:
            self._sa_scorer = SynthesizabilityScorer()
        return self._sa_scorer

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_prior_fp_bank(self, smiles: list[str]) -> int:
        """Install prior SMILES bank for the Morgan-FP soft floor. Returns bank size."""
        n = self.prior_fp_tether.set_bank(smiles)
        print(
            f"Prior FP tether: bank={n} floor={self.prior_fp_tether.floor}",
            flush=True,
        )
        return n

    def _score_locals(self, smiles_list: list[str], *, profile: bool) -> list[CO2LocalScore]:
        if self._pool is not None:
            n = len(smiles_list)
            if n == 0:
                return []
            # Chunk to cut per-task IPC overhead (important for fork/spawn alike).
            chunksize = max(1, n // (self.n_workers * 4))
            return list(
                self._pool.map(
                    _worker_score_co2_local, smiles_list, chunksize=chunksize
                )
            )
        return [
            score_co2_local(
                smi,
                self.calc,
                self.pfc,
                include_pfc=self.include_pfc,
                use_objective_ceilings=self.use_objective_ceilings,
                use_sigmoid_reward=self.use_sigmoid_reward,
                profile=profile,
            )
            for smi in smiles_list
        ]

    def compute_batch(self, smiles_list: list[str]) -> CO2RewardBatch:
        profile = _profile_rewards_enabled()
        t0 = time.perf_counter() if profile else 0.0

        t_div0 = time.perf_counter() if profile else 0.0
        diversity_scores = tanimoto_diversity_scores(smiles_list)
        t_div1 = time.perf_counter() if profile else 0.0

        t_local0 = time.perf_counter() if profile else 0.0
        locals_ = self._score_locals(smiles_list, profile=profile)
        t_local1 = time.perf_counter() if profile else 0.0

        t_asm0 = time.perf_counter() if profile else 0.0
        rewards: list[float] = []
        physchem_rewards: list[float] = []
        qualities: list[float] = []
        gate_ok: list[bool] = []
        mw_scales: list[float] = []

        ceil = self.use_objective_ceilings and not self.use_sigmoid_reward
        w_phys = _physchem_weight_sum(include_pfc=self.include_pfc)
        w_tot = w_phys + W_DIVERSE
        for idx, local in enumerate(locals_):
            if self.use_sigmoid_reward:
                div_u = term_score(
                    float(diversity_scores[idx]), "diversity", use_sigmoid=True
                )
            else:
                div_u = None
                diversity_term = W_DIVERSE * term_score(
                    float(diversity_scores[idx]),
                    "diversity",
                    use_ceilings=ceil,
                )
            if not local.gate_ok:
                rewards.append(float("nan"))
                physchem_rewards.append(float("nan"))
                qualities.append(float("nan"))
                gate_ok.append(False)
                mw_scales.append(float("nan"))
                continue

            scale = float(local.mw_scale)
            physchem_z = float(local.physchem_z)
            if self.use_sigmoid_reward:
                # Re-average physchem + diversity into (0, 1); soft axes add penalties.
                reward_u = (physchem_z * w_phys + float(div_u) * W_DIVERSE) / w_tot
                physchem = float(physchem_z)
                reward = float(reward_u)
            else:
                physchem = float(physchem_z)
                reward = float(physchem_z + diversity_term)

            can = local.can
            mol = Chem.MolFromSmiles(can) if can is not None else None
            # Keep MW pressure active in the no-penalties ablation.
            mw_ex = 0.0
            if mol is not None:
                mw_ex = mw_excess(
                    float(Descriptors.MolWt(mol)),
                    lo=MW_SCALE_LO,
                    hi=MW_SCALE_HI,
                    inner_lo=MW_FLAT_LO,
                    inner_hi=MW_FLAT_HI,
                )
            penalty = W_MW_PENALTY * mw_ex
            if self.apply_additive_terms:
                # Additive SA / long-arm / OH / NN / amine-floor / SMARTS.
                sa_ex = 0.0
                crot_ex = 0.0
                rot_win_ex = 0.0
                oh_ex = 0.0
                amine_floor_ex = 0.0
                epox_ex = 0.0
                sulfur_ex = 0.0
                ester_ex = 0.0
                acetal_ex = 0.0
                aminal_ex = 0.0
                hydrate_ex = 0.0
                hemiaminal_ex = 0.0
                peroxide_ex = 0.0
                peracid_ex = 0.0
                carboxylic_ex = 0.0
                amide_ex = 0.0
                hindered_ex = 0.0
                ring_o_ex = 0.0
                alkyne_ex = 0.0
                furan_ex = 0.0
                nonbasic_n_ex = 0.0
                nn_ex = nn_distance_excess(
                    float(local.min_nn_distance), thresh=NN_DIST_THRESH
                )
                if mol is not None and can is not None:
                    try:
                        sa = float(self._get_sa_scorer().score(can))
                    except Exception:
                        sa = SA_SCALE_ZERO_HI
                    sa_ex = high_end_excess(
                        sa, full_hi=SA_SCALE_FULL_HI, zero_hi=SA_SCALE_ZERO_HI
                    )
                    crot = max_consecutive_rotatable_bonds(mol)
                    crot_ex = high_end_excess(
                        float(crot),
                        full_hi=CONSEC_ROT_FULL_HI,
                        zero_hi=CONSEC_ROT_ZERO_HI,
                    )
                    n_rot = float(count_rotatable_bonds(mol))
                    rot_win_ex = rot_window_excess(
                        n_rot, lo=ROT_WINDOW_LO, hi=ROT_WINDOW_HI
                    )
                    n_am = float(count_amine_n(mol))
                    oh_ex = amine_justified_oh_excess(
                        float(count_oh(mol)),
                        n_am,
                        free=OH_AMINE_FREE,
                    )
                    amine_floor_ex = amine_floor_excess(n_am, floor=AMINE_FLOOR)
                    alerts = substructure_alert_excesses(
                        mol,
                        acetal_free=ACETAL_FREE,
                        sulfur_free=SULFUR_FREE,
                        ring_o_free=RING_O_FREE,
                    )
                    epox_ex = float(alerts["epoxide_aziridine"])
                    sulfur_ex = float(alerts["sulfur"])
                    ester_ex = float(alerts["ester"])
                    acetal_ex = float(alerts["cyclic_acetal"])
                    aminal_ex = float(alerts["aminal"])
                    hydrate_ex = float(alerts["hydrate"])
                    hemiaminal_ex = float(alerts["hemiaminal"])
                    peroxide_ex = float(alerts["peroxide"])
                    peracid_ex = float(alerts["peracid"])
                    carboxylic_ex = float(alerts["carboxylic_acid"])
                    amide_ex = float(alerts["amide"])
                    hindered_ex = float(alerts["hindered_amine_alpha"])
                    ring_o_ex = float(alerts["ring_oxygen"])
                    alkyne_ex = float(alerts["alkyne"])
                    furan_ex = float(alerts["furan"])
                    nonbasic_n_ex = float(nonbasic_n_excess(mol))
                penalty += (
                    W_SA_PENALTY * sa_ex
                    + W_CONSEC_ROT_PENALTY * crot_ex
                    + W_ROT_WINDOW_PENALTY * rot_win_ex
                    + W_OH_PENALTY * oh_ex
                    + W_NN_PENALTY * nn_ex
                    + W_AMINE_FLOOR_PENALTY * amine_floor_ex
                    + W_EPOXIDE_PENALTY * epox_ex
                    + W_SULFUR_PENALTY * sulfur_ex
                    + W_ESTER_PENALTY * ester_ex
                    + W_ACETAL_PENALTY * acetal_ex
                    + W_AMINAL_PENALTY * aminal_ex
                    + W_HYDRATE_PENALTY * hydrate_ex
                    + W_HEMIAMINAL_PENALTY * hemiaminal_ex
                    + W_PEROXIDE_PENALTY * peroxide_ex
                    + W_PERACID_PENALTY * peracid_ex
                    + W_CARBOXYLIC_ACID_PENALTY * carboxylic_ex
                    + W_AMIDE_PENALTY * amide_ex
                    + W_HINDERED_AMINE_PENALTY * hindered_ex
                    + W_RING_O_PENALTY * ring_o_ex
                    + W_ALKYNE_PENALTY * alkyne_ex
                    + W_FURAN_PENALTY * furan_ex
                    + W_NONBASIC_N_PENALTY * nonbasic_n_ex
                )

                if mol is not None and can is not None:
                    bonus = (
                        W_AZACROWN_BONUS * azacrown_bonus_units(mol)
                        + W_VICINAL_DIOL_BONUS * vicinal_diamine_diol(mol)
                    )
                    if bonus != 0.0:
                        reward += bonus
                        physchem += bonus

            if penalty != 0.0:
                reward -= penalty
                physchem -= penalty

            # Soft OOD floor vs prior bank (before quality / scaffold DF).
            if self.prior_fp_tether.enabled and can is not None:
                fp_factor = self.prior_fp_tether.factor(can)
                if fp_factor != 1.0:
                    reward *= fp_factor
                    physchem *= fp_factor

            quality = float(reward)

            if self.scaffold_filter is not None and mol is not None and can is not None:
                factor = self.scaffold_filter.factor(mol, can)
                self.scaffold_filter.remember(mol, can, reward)
                if factor != 1.0:
                    reward *= factor
                    physchem *= factor

            rewards.append(float(reward))
            physchem_rewards.append(float(physchem))
            qualities.append(quality)
            gate_ok.append(True)
            mw_scales.append(float(scale))
        t_asm1 = time.perf_counter() if profile else 0.0

        if profile:
            t_gate = sum(x.t_gate for x in locals_)
            t_mordred = sum(x.t_mordred for x in locals_)
            t_pfc = sum(x.t_pfc for x in locals_)
            n_gated = sum(1 for x in locals_ if x.gate_ok)
            print(
                "[MGW_REWARD_PROFILE] co2 "
                f"n={len(smiles_list)} gated={n_gated} workers={self.n_workers} "
                f"diversity_s={t_div1 - t_div0:.3f} "
                f"local_wall_s={t_local1 - t_local0:.3f} "
                f"gate_sum_s={t_gate:.3f} mordred_sum_s={t_mordred:.3f} "
                f"pfc_sum_s={t_pfc:.3f} "
                f"assemble_s={t_asm1 - t_asm0:.3f} "
                f"prior_fp_bank={self.prior_fp_tether.n_bank} "
                f"total_s={time.perf_counter() - t0:.3f}",
                flush=True,
            )

        return CO2RewardBatch(
            policy=rewards,
            raw_total=rewards,
            gate_ok=gate_ok,
            physchem=physchem_rewards,
            quality=qualities,
            mw_scale=mw_scales,
        )
