#!/usr/bin/env python3
"""Minimal asserts for CO2 explore / recalibration. Run: PYTHONPATH=. uv run python objectives/check_co2_rl_explore.py"""
from __future__ import annotations

import math
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import ConvexHull

from objectives.co2_polar_focal_concavity import (
    _DEFAULT_VDW,
    _VDW_RADII,
    PolarFocalConcavity,
    clear_pfc_cache,
    pfc_cache_size,
)
from objectives.co2_reward import (
    HAE_MULT_HI,
    HAE_MULT_LO,
    NN_DIST_THRESH,
    REF_STATS,
    W_APOL,
    W_DIVERSE,
    W_HAE_ALPHA,
    W_MW_PENALTY,
    W_NN_PENALTY,
    W_PFC_COMPOSITE,
    CO2ObjectiveScorer,
    OBJECTIVE_CEILINGS,
    RANGES,
    SIGMOID_K,
    sigmoid_key,
    sigmoid_to_unit,
    term_score,
    zscore_key,
)
from objectives.co2_rl_helpers import (
    EliteReplayBuffer,
    ScaffoldDiversityFilter,
    amine_floor_excess,
    amine_justified_oh_excess,
    baseline_relative_advantages,
    consec_rot_scale,
    count_alkyne,
    count_amine_n,
    count_aminal,
    count_amide,
    count_carboxylic_acid,
    count_cyclic_acetal,
    count_epoxide_aziridine,
    count_ester,
    count_furan,
    count_hemiaminal,
    count_hindered_amine_alpha,
    count_hydrate,
    count_oh,
    count_sulfonyl,
    azacrown_bonus_units,
    co2_gate_ok,
    max_ring_aliphatic_n,
    n_fragments,
    unique_gated_fraction,
    unique_canonical_learn_indices,
    valid_fraction,
    batch_health_metrics,
    count_peracid,
    count_peroxide,
    count_ring_oxygen,
    count_rotatable_bonds,
    count_sulfur,
    decline_streak_should_restore,
    early_stop_should_stop,
    excess_from_scale,
    explore_schedule_progress,
    gate_frac_streak_should_restore,
    hae_multiplier,
    hae_raw,
    high_end_excess,
    inception_replay_k,
    linear_schedule,
    max_consecutive_rotatable_bonds,
    min_nonbonded_nn_distance,
    murcko_scaffold_smiles,
    mw_excess,
    mw_scale,
    nn_distance_excess,
    nonbasic_n_excess,
    oh_scale,
    pad_credit_mask,
    per_heavy_atom,
    rot_window_excess,
    sa_scale,
    should_skip_thin_batch,
    substructure_alert_excesses,
    top_quantile_indices,
    update_ema,
    vicinal_diamine_diol,
    whiten_advantages,
)

# Gated CO2 fixture: HA>=10, >=2 aliphatic amines, MW in flat band.
DIAMINE = "NCCCCCCCCCCN"

from transpharmer.SMILEValidifier import valid_smiles, validate_smiles


def _voxel_ratio_ref(positions: np.ndarray, atomic_nums: np.ndarray, step: float) -> float:
    """Original pure-Python voxel ratio (parity oracle for vectorized path)."""
    n = positions.shape[0]
    if n < 4:
        return 0.0
    try:
        hull = ConvexHull(positions)
    except Exception:
        return 0.0
    eqs = hull.equations
    radii = np.array([_VDW_RADII.get(int(z), _DEFAULT_VDW) for z in atomic_nums], dtype=float)
    lo = positions.min(axis=0) - step
    hi = positions.max(axis=0) + step
    hull_voxels = 0
    hollow_voxels = 0
    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            z = z0
            while z < z1:
                pt = np.array([x, y, z], dtype=float)
                if np.any(eqs[:, :3] @ pt + eqs[:, 3] > 1e-10):
                    z += step
                    continue
                hull_voxels += 1
                dists = np.linalg.norm(positions - pt, axis=1)
                if np.any(dists <= radii):
                    z += step
                    continue
                hollow_voxels += 1
                z += step
            y += step
        x += step
    return 0.0 if hull_voxels == 0 else hollow_voxels / hull_voxels


def main() -> None:
    assert W_APOL == W_DIVERSE == 1.0
    assert W_PFC_COMPOSITE == 1.25

    # SMILES validity API: singular vs list filter
    assert valid_smiles("CCO") is True
    assert valid_smiles("CCO)C1") is False
    try:
        validate_smiles("CCO")
        raise AssertionError("validate_smiles(str) should raise TypeError")
    except TypeError:
        pass
    assert validate_smiles(["CCO", "CCO)C1"]) == ["CCO"]

    scorer = CO2ObjectiveScorer(random_seed=42, n_conformers=2, use_scaffold_filter=False)

    fail = scorer.compute_batch(["C[N+](C)(C)C"])
    assert not fail.gate_ok[0]
    assert math.isnan(fail.policy[0]), fail.policy[0]

    bad = scorer.compute_batch(["not_a_smiles"])
    assert not bad.gate_ok[0]
    assert math.isnan(bad.policy[0])

    # Charge/atoms gate: benzene passes; charged fails above.
    ok = scorer.compute_batch(["c1ccccc1"])
    assert bool(ok.gate_ok[0]) is True, (ok.gate_ok[0], ok.policy[0])
    assert math.isfinite(ok.policy[0])
    assert ok.mw_scale[0] == 1.0 or math.isfinite(ok.mw_scale[0])

    # No-PFC path: skips conformer/PFC work; physchem still finite.
    scorer_nopf = CO2ObjectiveScorer(
        random_seed=42, n_conformers=2, use_scaffold_filter=False, include_pfc=False
    )
    assert scorer_nopf.pfc is None
    nopf = scorer_nopf.compute_batch(["c1ccccc1O"])
    assert bool(nopf.gate_ok[0]) and math.isfinite(nopf.policy[0])
    scorer_nopf.close()

    # Mid-slope at range midpoints (RANGES = GuacaMol∪AxelRolov envelope)
    assert SIGMOID_K == 16.0
    assert RANGES["diversity"] == (0.0, 1.0)
    from objectives.co2_reward import RANGE_PAD, _BASE_RANGES, pad_range

    assert RANGE_PAD == 0.05
    assert RANGES["apol"] == pad_range(*_BASE_RANGES["apol"])
    mid_apol = sum(RANGES["apol"]) / 2.0
    mid_logp = sum(RANGES["LogP"]) / 2.0
    apol_s = sigmoid_to_unit(mid_apol, *RANGES["apol"])
    logp_s = sigmoid_to_unit(mid_logp, *RANGES["LogP"])
    assert 0.3 <= apol_s <= 0.7, apol_s
    assert 0.3 <= logp_s <= 0.7, logp_s
    # Sharper k discriminates p10/p90 more than k=8
    lo, hi = RANGES["apol"]
    p10_k16 = sigmoid_to_unit(lo, lo, hi, k=16.0)
    p90_k16 = sigmoid_to_unit(hi, lo, hi, k=16.0)
    p10_k8 = sigmoid_to_unit(lo, lo, hi, k=8.0)
    p90_k8 = sigmoid_to_unit(hi, lo, hi, k=8.0)
    assert (p90_k16 - p10_k16) > (p90_k8 - p10_k8)

    div_sig = sigmoid_to_unit(1.0, *RANGES["diversity"])
    assert 0.0 < div_sig <= 1.0

    # Soft-bounded z-scores: +5σ stays below Z_SOFT=3, still > old 0.625 ceiling
    assert abs(zscore_key(REF_STATS["apol"][0], "apol")) < 1e-6
    z5 = zscore_key(REF_STATS["apol"][0] + 5 * REF_STATS["apol"][1], "apol")
    assert 0.625 < z5 < 3.0, z5
    assert "HAE" in REF_STATS and "HAE" in RANGES
    assert abs(zscore_key(REF_STATS["HAE"][0], "HAE")) < 1e-5

    # HAE helpers kept; scoring no longer multiplies physchem by hae_multiplier.
    assert per_heavy_atom(10.0, 5) == 2.0
    assert per_heavy_atom(10.0, 0) == 10.0
    assert hae_raw(1.0, 1.0, 10, pfc_u=1.0) > hae_raw(1.0, 1.0, 20, pfc_u=1.0)
    m_hi = hae_multiplier(2.0, alpha=W_HAE_ALPHA, lo=HAE_MULT_LO, hi=HAE_MULT_HI)
    m_lo = hae_multiplier(-2.0, alpha=W_HAE_ALPHA, lo=HAE_MULT_LO, hi=HAE_MULT_HI)
    assert HAE_MULT_LO <= m_lo < 1.0 < m_hi <= HAE_MULT_HI
    assert m_hi > m_lo
    # Clip extremes
    assert hae_multiplier(100.0) == HAE_MULT_HI
    assert hae_multiplier(-100.0) == HAE_MULT_LO

    # Objective ceilings: extreme raw matches edge; unlimited scores further on LogP (−) / basicity
    assert OBJECTIVE_CEILINGS == RANGES
    lo_logp, _hi_logp = OBJECTIVE_CEILINGS["LogP"]
    _lo_bas, hi_bas = OBJECTIVE_CEILINGS["basicity"]
    z_logp_extreme = zscore_key(-20.0, "LogP", use_ceilings=True)
    z_logp_edge = zscore_key(lo_logp, "LogP", use_ceilings=True)
    assert abs(z_logp_extreme - z_logp_edge) < 1e-12
    z_logp_free = zscore_key(-20.0, "LogP", use_ceilings=False)
    # Inverted LogP term: lower LogP → higher contribution when unlimited
    assert (-z_logp_free) > (-z_logp_extreme)
    z_bas_extreme = zscore_key(200.0, "basicity", use_ceilings=True)
    z_bas_edge = zscore_key(hi_bas, "basicity", use_ceilings=True)
    assert abs(z_bas_extreme - z_bas_edge) < 1e-12
    z_bas_free = zscore_key(200.0, "basicity", use_ceilings=False)
    assert z_bas_free > z_bas_extreme

    scorer_ceil = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        include_pfc=False,
        use_objective_ceilings=True,
    )
    scorer_free = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        include_pfc=False,
        use_objective_ceilings=False,
    )
    # Diamine: should gate ok under both
    for s in (scorer_ceil, scorer_free):
        b = s.compute_batch([DIAMINE])
        assert bool(b.gate_ok[0]) and math.isfinite(b.policy[0])
    scorer_ceil.close()
    scorer_free.close()

    # Sigmoid reward: terms in (0,1); extreme basicity saturates near hi anchor
    assert abs(sigmoid_key(RANGES["basicity"][1], "basicity") - sigmoid_key(200.0, "basicity")) < 1e-3
    mid_bas = sigmoid_key(sum(RANGES["basicity"]) / 2.0, "basicity")
    assert 0.4 < mid_bas < 0.6, mid_bas
    scorer_sig = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        include_pfc=False,
        use_sigmoid_reward=True,
    )
    assert scorer_sig.use_sigmoid_reward
    sig_batch = scorer_sig.compute_batch([DIAMINE, "NCCNCCCCCCCCN"])
    assert all(sig_batch.gate_ok) and all(math.isfinite(x) for x in sig_batch.policy)
    # Mid-range molecules: sigmoid base in (0, 1); additive MW/OH penalties may nudge slightly.
    assert all(math.isfinite(x) for x in sig_batch.policy), sig_batch.policy
    assert all(math.isfinite(x) for x in sig_batch.physchem), sig_batch.physchem
    scorer_sig.close()
    assert abs(term_score(54.15, "apol", use_sigmoid=True) - sigmoid_to_unit(54.15, *RANGES["apol"])) < 1e-12

    scorer_z = CO2ObjectiveScorer(
        random_seed=42, n_conformers=2, use_scaffold_filter=False
    )
    batch = scorer_z.compute_batch([DIAMINE, "NCCNCCCCCCCCN", "not_a_smiles"])
    assert len(batch.physchem) == len(batch.policy) == 3
    assert math.isnan(batch.physchem[2]) and math.isnan(batch.policy[2])
    for i in (0, 1):
        assert math.isfinite(batch.physchem[i]) and math.isfinite(batch.policy[i])
    diam = scorer_z.compute_batch([DIAMINE])
    assert math.isfinite(diam.physchem[0])
    # Extreme raw still contributes above old unit ceiling after soft-bound
    assert W_APOL * zscore_key(100.0, "apol") > 0.625
    assert zscore_key(100.0, "apol") <= 3.0

    # Scaffold diversity filter: bucket fills → penalty (phenol has a Murcko ring)
    df = ScaffoldDiversityFilter(bucket_size=3, penalty=0.0)
    mol = Chem.MolFromSmiles("c1ccccc1O")
    can = Chem.MolToSmiles(mol)
    assert murcko_scaffold_smiles(mol) is not None
    for _ in range(3):
        assert df.factor(mol, can) == 1.0
        df.remember(mol, f"{can}_{_}", 1.0)  # distinct smiles, same scaffold
    assert df.factor(mol, "c1ccccc1N") == 0.0

    scorer_df = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        scaffold_bucket=2,
        scaffold_penalty=0.0,
        use_scaffold_filter=True,
    )
    # Same Murcko (benzene) diamines past bucket → later rewards zeroed
    smiles_rep = ["NCc1ccc(CN)cc1", "NCc1ccc(C(C)N)cc1", "NCc1cc(F)c(CN)cc1"]
    bdf = scorer_df.compute_batch(smiles_rep)
    gated = [i for i, ok in enumerate(bdf.gate_ok) if ok]
    assert len(gated) >= 2
    # At least one later gated molecule should be penalized to 0 if same Murcko
    scafs = [murcko_scaffold_smiles(Chem.MolFromSmiles(smiles_rep[i])) for i in gated]
    if len(set(scafs)) == 1 and len(gated) > 2:
        assert bdf.policy[gated[-1]] == 0.0

    # Prior Morgan-FP soft floor: bank hit → factor 1; far SMILES scaled.
    from objectives.prior_fp_tether import prior_fp_floor_factor

    assert abs(prior_fp_floor_factor(0.075, 0.15) - 0.5) < 1e-12
    scorer_fp = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        include_pfc=False,
        prior_fp_floor=0.15,
    )
    scorer_fp.set_prior_fp_bank([DIAMINE, "NCCNCCCCCCCCN", "NCc1ccc(CN)cc1"])
    assert scorer_fp.prior_fp_tether.factor(DIAMINE) == 1.0
    far_factor = scorer_fp.prior_fp_tether.factor("C")
    assert 0.0 <= far_factor < 1.0, far_factor
    # Reward path: identical molecule with/without tether — bank hit unchanged.
    base = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        include_pfc=False,
        prior_fp_floor=0.0,
    )
    b0 = base.compute_batch([DIAMINE])
    b1 = scorer_fp.compute_batch([DIAMINE])
    assert b0.gate_ok[0] and b1.gate_ok[0]
    assert abs(b0.policy[0] - b1.policy[0]) < 1e-9, (b0.policy[0], b1.policy[0])
    scorer_fp.close()
    base.close()

    # Elite replay keeps best quality, drops worst when over capacity
    buf = EliteReplayBuffer(capacity=3)
    buf.offer("CCO", 1.0)
    buf.offer("c1ccccc1", 5.0)
    buf.offer("CCN", 2.0)
    buf.offer("CCC", 4.0)  # evicts CCO
    assert len(buf) == 3
    assert "CCO" not in dict(buf.to_list())
    assert dict(buf.to_list())["c1ccccc1"] == 5.0
    samp = buf.sample(2, np.random.default_rng(0))
    assert len(samp) == 2
    # Quality-weighted sample prefers the top molecule
    buf_w = EliteReplayBuffer(capacity=4)
    buf_w.offer("low", 1.0)
    buf_w.offer("mid", 2.0)
    buf_w.offer("high", 100.0)
    hits = sum(
        1
        for _ in range(40)
        if buf_w.sample(1, np.random.default_rng(_), weighted=True)[0][0] == "high"
    )
    assert hits >= 25, hits
    assert inception_replay_k(8, 32, boost_active=False, n_on_policy=40, max_frac=0.5) == 8
    assert inception_replay_k(8, 32, boost_active=True, n_on_policy=40, max_frac=0.5) == 32
    # frac cap: elite/(on+elite)<=0.5 → elite<=on
    assert inception_replay_k(8, 32, boost_active=True, n_on_policy=10, max_frac=0.5) == 10
    assert inception_replay_k(8, 32, boost_active=True, n_on_policy=0, max_frac=0.5) == 0

    assert abs(mw_scale(275.0) - 1.0) < 1e-9
    assert abs(mw_scale(100.0) - 1.0) < 1e-9
    assert abs(mw_scale(350.0) - 1.0) < 1e-9
    assert abs(mw_scale(50.0)) < 1e-9
    assert abs(mw_scale(75.0) - 0.5) < 1e-9
    assert abs(mw_scale(400.0)) < 1e-9
    assert abs(mw_scale(450.0)) < 1e-9
    assert abs(mw_scale(600.0)) < 1e-9
    assert mw_scale(360.0) > mw_scale(380.0) > 0.0
    assert mw_scale(90.0) > mw_scale(60.0) > 0.0

    assert abs(sa_scale(3.0) - 1.0) < 1e-9
    assert abs(sa_scale(3.5) - 1.0) < 1e-9
    assert abs(sa_scale(4.25) - 0.5) < 1e-9
    assert abs(sa_scale(5.0)) < 1e-9
    assert abs(sa_scale(6.0)) < 1e-9

    assert abs(consec_rot_scale(3) - 1.0) < 1e-9
    assert abs(consec_rot_scale(4) - 1.0) < 1e-9
    assert abs(consec_rot_scale(5.5) - 0.5) < 1e-9
    assert abs(consec_rot_scale(7)) < 1e-9
    assert abs(consec_rot_scale(10)) < 1e-9

    # Total rotatable-bond window (ideal 3–6): excess outside.
    assert abs(rot_window_excess(3)) < 1e-9
    assert abs(rot_window_excess(6)) < 1e-9
    assert abs(rot_window_excess(4)) < 1e-9
    assert abs(rot_window_excess(2) - 1.0) < 1e-9
    assert abs(rot_window_excess(1) - 2.0) < 1e-9
    assert abs(rot_window_excess(0) - 3.0) < 1e-9
    assert abs(rot_window_excess(7) - 1.0) < 1e-9
    assert abs(rot_window_excess(8) - 2.0) < 1e-9

    assert abs(oh_scale(2) - 1.0) < 1e-9
    assert abs(oh_scale(3) - 1.0) < 1e-9
    assert abs(oh_scale(4) - 0.5) < 1e-9
    assert abs(oh_scale(5)) < 1e-9
    assert abs(oh_scale(10)) < 1e-9
    assert abs(excess_from_scale(1.0)) < 1e-9
    assert abs(excess_from_scale(0.0) - 1.0) < 1e-9
    assert abs(excess_from_scale(0.5) - 0.5) < 1e-9
    # Uncapped excess continues past the unit/outer anchor.
    assert abs(high_end_excess(3, full_hi=3.0, zero_hi=5.0)) < 1e-9
    assert abs(high_end_excess(4, full_hi=3.0, zero_hi=5.0) - 0.5) < 1e-9
    assert abs(high_end_excess(5, full_hi=3.0, zero_hi=5.0) - 1.0) < 1e-9
    assert abs(high_end_excess(7, full_hi=3.0, zero_hi=5.0) - 2.0) < 1e-9
    assert high_end_excess(14, full_hi=3.0, zero_hi=5.0) > high_end_excess(
        5, full_hi=3.0, zero_hi=5.0
    )
    assert abs(mw_excess(225.0)) < 1e-9
    assert abs(mw_excess(75.0) - 0.5) < 1e-9  # midway lo→inner_lo
    assert abs(mw_excess(50.0) - 1.0) < 1e-9
    assert mw_excess(25.0) > 1.0  # continues below lo
    assert mw_excess(450.0) > 1.0  # continues above hi
    phenol = Chem.MolFromSmiles("c1ccc(O)cc1")
    glucose_like = Chem.MolFromSmiles("OCC1OC(O)C(O)C(O)C1O")
    assert phenol is not None and glucose_like is not None
    assert count_oh(phenol) == 1
    assert count_oh(glucose_like) >= 5
    assert oh_scale(float(count_oh(glucose_like))) == 0.0
    assert count_amine_n(Chem.MolFromSmiles("NCCO")) == 1  # MEA
    assert count_amine_n(glucose_like) == 0
    assert abs(amine_justified_oh_excess(1, 1, free=0.0)) < 1e-9  # MEA
    assert abs(amine_justified_oh_excess(2, 1, free=0.0) - 1.0) < 1e-9  # DEA-ish
    assert abs(amine_justified_oh_excess(3, 2, free=0.0) - 1.0) < 1e-9  # 3 OH / 2 amine
    assert abs(amine_justified_oh_excess(14, 0, free=0.0) - 14.0) < 1e-9  # polyol

    # Long linear arm → high consecutive rot; short branched hinges stay lower.
    arm = Chem.MolFromSmiles("CCCCCCCCCCCC")  # n-dodecane
    branched = Chem.MolFromSmiles("CC(C)C(C)C(C)C")  # short isopropyl chain
    assert arm is not None and branched is not None
    n_arm = max_consecutive_rotatable_bonds(arm)
    n_br = max_consecutive_rotatable_bonds(branched)
    assert n_arm > 4, n_arm
    assert n_arm > n_br, (n_arm, n_br)
    assert consec_rot_scale(float(n_arm)) < 1.0

    # MW>400: scale 0; uncapped mw_excess > 1.
    heavy = "C" * 40  # C40H82 ≈ 563 Da
    from rdkit.Chem import Descriptors as _Desc

    heavy_mol = Chem.MolFromSmiles(heavy)
    assert heavy_mol is not None and _Desc.MolWt(heavy_mol) > 400.0
    heavy_b = scorer_z.compute_batch([heavy])
    assert heavy_b.gate_ok[0] is True
    assert abs(heavy_b.mw_scale[0]) < 1e-9
    assert mw_excess(_Desc.MolWt(heavy_mol)) > 1.0
    assert math.isfinite(heavy_b.policy[0])
    assert math.isfinite(heavy_b.physchem[0])

    # Mid-band molecule keeps scale 1 (naphthalene MW≈128)
    mid_b = scorer_z.compute_batch(["c1ccc2ccccc2c1"])
    assert mid_b.gate_ok[0] is True
    assert abs(mid_b.mw_scale[0] - 1.0) < 1e-9
    assert math.isfinite(mid_b.policy[0])
    # Phenol MW≈94 is on the low ramp → partial MW penalty only.
    phenol_b = scorer_z.compute_batch(["c1ccccc1O"])
    assert phenol_b.gate_ok[0] is True
    assert 0.0 < phenol_b.mw_scale[0] < 1.0
    assert math.isfinite(phenol_b.policy[0])
    mw_pen = W_MW_PENALTY * mw_excess(_Desc.MolWt(Chem.MolFromSmiles("c1ccccc1O")))
    assert 0.0 < mw_pen < W_MW_PENALTY
    assert 0.0 < mw_scale(_Desc.MolWt(Chem.MolFromSmiles("c1ccccc1O"))) < 1.0

    # MGW_NO_PENALTIES path: retain MW penalty, skip other penalties / bonuses.
    scorer_nopen = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        include_pfc=False,
        apply_additive_terms=False,
    )
    scorer_pen = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        include_pfc=False,
        apply_additive_terms=True,
    )
    nopen_b = scorer_nopen.compute_batch(["c1ccccc1O"])
    pen_b = scorer_pen.compute_batch(["c1ccccc1O"])
    assert nopen_b.gate_ok[0] and pen_b.gate_ok[0]
    assert nopen_b.policy[0] > pen_b.policy[0]
    scorer_nopen.close()
    scorer_pen.close()

    # Long-arm molecule: consec-rot + total-rot window both fire; still gated.
    long_arm_smi = "CCOCCOCCOCCOCCOCCO"
    long_mol = Chem.MolFromSmiles(long_arm_smi)
    assert long_mol is not None
    assert max_consecutive_rotatable_bonds(long_mol) > 4
    n_rot_long = count_rotatable_bonds(long_mol)
    assert n_rot_long > 6
    assert rot_window_excess(float(n_rot_long)) > 0.0
    long_b = scorer_z.compute_batch([long_arm_smi])
    assert long_b.gate_ok[0] is True
    assert long_b.mw_scale[0] > 0.0
    assert math.isfinite(long_b.policy[0])
    assert consec_rot_scale(float(max_consecutive_rotatable_bonds(long_mol))) < 1.0

    # Rigid low-rot molecule still gate_ok with window excess on the low side.
    rigid_smi = "c1ccccc1"
    rigid_mol = Chem.MolFromSmiles(rigid_smi)
    assert rigid_mol is not None
    n_rot_rigid = count_rotatable_bonds(rigid_mol)
    assert n_rot_rigid < 3
    assert rot_window_excess(float(n_rot_rigid)) > 0.0
    rigid_b = scorer_z.compute_batch([rigid_smi])
    assert rigid_b.gate_ok[0] is True
    assert math.isfinite(rigid_b.policy[0])

    # Polyol: amine-justified OH excess large; MEA-like stays cheap.
    polyol = "C1(O)C(CO)OC(OC2C(O)C(O)C(O)C(CO)O2)C(O)C1O"
    poly_mol = Chem.MolFromSmiles(polyol)
    assert poly_mol is not None and count_oh(poly_mol) >= 5
    oh_ex = amine_justified_oh_excess(
        float(count_oh(poly_mol)), float(count_amine_n(poly_mol)), free=0.0
    )
    assert oh_ex >= 5.0
    mea = Chem.MolFromSmiles("NCCO")
    assert mea is not None
    assert (
        amine_justified_oh_excess(
            float(count_oh(mea)), float(count_amine_n(mea)), free=0.0
        )
        == 0.0
    )
    poly_b = scorer_z.compute_batch([polyol])
    assert poly_b.gate_ok[0] is True
    assert math.isfinite(poly_b.policy[0])

    # Min non-bonded N···N: bonded hydrazine does not trigger; close through-space does.
    assert nn_distance_excess(float("nan"), thresh=NN_DIST_THRESH) == 0.0
    assert nn_distance_excess(3.5, thresh=NN_DIST_THRESH) == 0.0
    assert abs(nn_distance_excess(1.4, thresh=NN_DIST_THRESH) - (2.8 - 1.4) / 2.8) < 1e-12
    hyd = Chem.AddHs(Chem.MolFromSmiles("CN(C)N(C)C"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    assert AllChem.EmbedMolecule(hyd, params) == 0
    # Hydrazine N–N is bonded (graph dist 1) → no qualifying pair → NaN → excess 0
    d_hyd = min_nonbonded_nn_distance(hyd, hyd.GetConformer())
    assert math.isnan(d_hyd)
    assert nn_distance_excess(d_hyd) == 0.0
    # Hand-place two N at graph dist ≥ 4 with Euclidean 2.0 Å
    nn_mol = Chem.AddHs(Chem.MolFromSmiles("NCCCN"))
    conf = Chem.Conformer(nn_mol.GetNumAtoms())
    n_ids = [a.GetIdx() for a in nn_mol.GetAtoms() if a.GetAtomicNum() == 7]
    assert len(n_ids) == 2
    conf.SetAtomPosition(n_ids[0], (0.0, 0.0, 0.0))
    conf.SetAtomPosition(n_ids[1], (2.0, 0.0, 0.0))
    for i in range(nn_mol.GetNumAtoms()):
        if i not in n_ids:
            conf.SetAtomPosition(i, (0.5, 0.5, 0.5))
    nn_mol.AddConformer(conf, assignId=True)
    d_close = min_nonbonded_nn_distance(nn_mol, nn_mol.GetConformer())
    assert abs(d_close - 2.0) < 1e-6
    assert nn_distance_excess(d_close, thresh=NN_DIST_THRESH) > 0.0
    assert W_NN_PENALTY == 1.0

    # Soft amine floor: 0 amines → 1.0 excess; ≥3 → 0; still gated.
    from objectives.co2_reward import AMINE_FLOOR, W_AMINE_FLOOR_PENALTY

    assert AMINE_FLOOR == 3.0 and W_AMINE_FLOOR_PENALTY == 1.0
    assert abs(amine_floor_excess(0.0, floor=3.0) - 1.0) < 1e-12
    assert abs(amine_floor_excess(1.0, floor=3.0) - 2.0 / 3.0) < 1e-12
    assert abs(amine_floor_excess(2.0, floor=3.0) - 1.0 / 3.0) < 1e-12
    assert amine_floor_excess(3.0, floor=3.0) == 0.0
    assert amine_floor_excess(5.0, floor=3.0) == 0.0
    benzene_b = scorer_z.compute_batch(["c1ccccc1"])
    diam_b = scorer_z.compute_batch([DIAMINE])  # 2 amines → partial floor penalty
    assert benzene_b.gate_ok[0] and diam_b.gate_ok[0]
    # Phenol/benzene have 0 aliphatic amines → full floor penalty vs a 3+ amine mol
    assert math.isfinite(benzene_b.policy[0])

    # Soft SMARTS alerts: counts / excesses; still gated (soft only).
    from objectives.co2_reward import (
        ACETAL_FREE,
        SULFUR_FREE,
        W_ACETAL_PENALTY,
        W_EPOXIDE_PENALTY,
        W_ESTER_PENALTY,
        W_SULFUR_PENALTY,
    )

    assert W_EPOXIDE_PENALTY == W_SULFUR_PENALTY == W_ESTER_PENALTY == W_ACETAL_PENALTY == 1.0
    assert ACETAL_FREE == 1.0 and SULFUR_FREE == 0.0
    epox = Chem.MolFromSmiles("C1OC1")
    azir = Chem.MolFromSmiles("C1NC1")
    alk = Chem.MolFromSmiles("CCCC")
    assert count_epoxide_aziridine(epox) >= 1 and count_epoxide_aziridine(azir) >= 1
    assert count_epoxide_aziridine(alk) == 0
    assert substructure_alert_excesses(epox)["epoxide_aziridine"] >= 1.0
    assert substructure_alert_excesses(alk)["epoxide_aziridine"] == 0.0
    thio = Chem.MolFromSmiles("CCSCC")
    assert count_sulfur(thio) == 1 and count_sulfur(alk) == 0
    assert substructure_alert_excesses(thio)["sulfur"] == 1.0
    est = Chem.MolFromSmiles("CC(=O)OC")
    acid = Chem.MolFromSmiles("CC(=O)O")
    assert count_ester(est) >= 1 and count_ester(acid) == 0
    diox = Chem.MolFromSmiles("C1COCO1")
    bis = Chem.MolFromSmiles("C1OC(CO1)C2OCCO2")
    assert count_cyclic_acetal(diox) == 1
    assert substructure_alert_excesses(diox, acetal_free=1.0)["cyclic_acetal"] == 0.0
    assert count_cyclic_acetal(bis) >= 2
    assert substructure_alert_excesses(bis, acetal_free=1.0)["cyclic_acetal"] >= 1.0
    sugar = Chem.MolFromSmiles("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
    assert count_cyclic_acetal(sugar) == 0
    # Soft only: alert molecules still gate_ok
    for smi in ("C1OC1", "CCSCC", "CC(=O)OC", "C1COCO1"):
        b = scorer_z.compute_batch([smi])
        assert b.gate_ok[0] is True and math.isfinite(b.policy[0])

    # Aminal / hydrate / hemiaminal + non-basic N (soft only).
    from objectives.co2_reward import (
        W_AMINAL_PENALTY,
        W_HEMIAMINAL_PENALTY,
        W_HYDRATE_PENALTY,
        W_NONBASIC_N_PENALTY,
    )

    assert (
        W_AMINAL_PENALTY
        == W_HYDRATE_PENALTY
        == W_HEMIAMINAL_PENALTY
        == W_NONBASIC_N_PENALTY
        == 1.0
    )
    aminal = Chem.MolFromSmiles("CC(N)(N)C")
    hydrate = Chem.MolFromSmiles("CC(O)(O)C")
    hemi = Chem.MolFromSmiles("CC(N)(O)C")
    assert aminal is not None and hydrate is not None and hemi is not None
    assert count_aminal(aminal) >= 1
    assert substructure_alert_excesses(aminal)["aminal"] >= 1.0
    assert count_hydrate(hydrate) >= 1
    assert substructure_alert_excesses(hydrate)["hydrate"] >= 1.0
    assert count_hemiaminal(hemi) >= 1
    assert substructure_alert_excesses(hemi)["hemiaminal"] >= 1.0
    assert substructure_alert_excesses(alk)["aminal"] == 0.0
    mea = Chem.MolFromSmiles("NCCO")
    assert mea is not None and nonbasic_n_excess(mea) == 0.0
    pyr = Chem.MolFromSmiles("c1ccncc1")
    amide = Chem.MolFromSmiles("CC(=O)N")
    assert pyr is not None and amide is not None
    assert nonbasic_n_excess(pyr) >= 1.0
    assert nonbasic_n_excess(amide) >= 1.0
    for smi in ("CC(N)(N)C", "CC(O)(O)C", "CC(N)(O)C", "c1ccncc1", "CC(=O)N"):
        b = scorer_z.compute_batch([smi])
        assert b.gate_ok[0] is True and math.isfinite(b.policy[0])

    # Peroxide / peracid / carboxylic acid (ester already split from acid).
    from objectives.co2_reward import (
        W_CARBOXYLIC_ACID_PENALTY,
        W_PERACID_PENALTY,
        W_PEROXIDE_PENALTY,
    )

    assert (
        W_PEROXIDE_PENALTY
        == W_PERACID_PENALTY
        == W_CARBOXYLIC_ACID_PENALTY
        == 1.0
    )
    hydro = Chem.MolFromSmiles("COO")
    dialkyl = Chem.MolFromSmiles("COOC")
    ether = Chem.MolFromSmiles("CCOCC")
    acetal = Chem.MolFromSmiles("C1COCO1")
    perac = Chem.MolFromSmiles("CC(=O)OO")
    assert hydro is not None and dialkyl is not None
    assert count_peroxide(hydro) >= 1 and count_peroxide(dialkyl) >= 1
    assert substructure_alert_excesses(hydro)["peroxide"] >= 1.0
    assert substructure_alert_excesses(dialkyl)["carboxylic_acid"] == 0.0
    assert count_peroxide(ether) == 0 and count_peroxide(acetal) == 0
    assert perac is not None
    assert count_peroxide(perac) >= 1 and count_peracid(perac) >= 1
    assert count_carboxylic_acid(perac) == 0
    assert count_carboxylic_acid(acid) >= 1 and count_ester(acid) == 0
    assert count_ester(est) >= 1 and count_carboxylic_acid(est) == 0
    for smi in ("COO", "COOC", "CC(=O)OO", "CC(=O)O", "CC(=O)OC"):
        b = scorer_z.compute_batch([smi])
        assert b.gate_ok[0] is True and math.isfinite(b.policy[0])

    # Amide / hindered amine α-carbon / ring-oxygen cap (soft only).
    from objectives.co2_reward import (
        RING_O_FREE,
        W_AMIDE_PENALTY,
        W_HINDERED_AMINE_PENALTY,
        W_RING_O_PENALTY,
    )

    assert W_AMIDE_PENALTY == W_HINDERED_AMINE_PENALTY == W_RING_O_PENALTY == 1.0
    assert RING_O_FREE == 1.0
    acetamide = Chem.MolFromSmiles("CC(=O)N")
    ethylamine = Chem.MolFromSmiles("CCN")
    assert acetamide is not None and ethylamine is not None
    assert count_amide(acetamide) >= 1 and count_amide(ethylamine) == 0
    assert substructure_alert_excesses(acetamide)["amide"] >= 1.0
    assert substructure_alert_excesses(ethylamine)["amide"] == 0.0
    ipr = Chem.MolFromSmiles("CC(C)N")
    tbu = Chem.MolFromSmiles("CC(C)(C)N")
    npr = Chem.MolFromSmiles("CCCN")
    assert ipr is not None and tbu is not None and npr is not None
    assert count_hindered_amine_alpha(ipr) == 1
    assert count_hindered_amine_alpha(tbu) == 1
    assert count_hindered_amine_alpha(npr) == 0
    assert substructure_alert_excesses(npr)["hindered_amine_alpha"] == 0.0
    # Amide with branched N-alkyl should not fire hindered (amide excluded).
    assert count_hindered_amine_alpha(Chem.MolFromSmiles("CC(=O)NC(C)C")) == 0
    glucose = Chem.MolFromSmiles("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
    sucrose = Chem.MolFromSmiles(
        "OC[C@H]1O[C@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
    )
    morph = Chem.MolFromSmiles("C1COCCN1")
    assert glucose is not None and sucrose is not None and morph is not None
    assert count_ring_oxygen(glucose) == 1
    assert substructure_alert_excesses(glucose, ring_o_free=1.0)["ring_oxygen"] == 0.0
    assert count_ring_oxygen(morph) == 1
    assert substructure_alert_excesses(morph, ring_o_free=1.0)["ring_oxygen"] == 0.0
    assert count_ring_oxygen(sucrose) >= 2
    assert substructure_alert_excesses(sucrose, ring_o_free=1.0)["ring_oxygen"] >= 1.0
    for smi in (
        "CC(=O)N",
        "CCN",
        "CC(C)N",
        "CC(C)(C)N",
        "CCCN",
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "C1COCCN1",
    ):
        b = scorer_z.compute_batch([smi])
        assert b.gate_ok[0] is True and math.isfinite(b.policy[0])

    # Alkyne / aromatic furan (soft only). OH knobs: FREE=0, W=2.
    from objectives.co2_reward import (
        OH_AMINE_FREE,
        W_ALKYNE_PENALTY,
        W_FURAN_PENALTY,
        W_OH_PENALTY,
    )

    assert OH_AMINE_FREE == 0.0 and W_OH_PENALTY == 2.0
    assert W_ALKYNE_PENALTY == 2.0 and W_FURAN_PENALTY == 1.0
    butyne = Chem.MolFromSmiles("CC#CC")
    phac = Chem.MolFromSmiles("C#Cc1ccccc1")
    alkane = Chem.MolFromSmiles("CCCC")
    nitrile = Chem.MolFromSmiles("CCC#N")
    assert butyne is not None and phac is not None
    assert count_alkyne(butyne) >= 1 and count_alkyne(phac) >= 1
    assert count_alkyne(alkane) == 0 and count_alkyne(nitrile) == 0
    assert substructure_alert_excesses(butyne)["alkyne"] >= 1.0
    assert substructure_alert_excesses(alkane)["alkyne"] == 0.0
    mfur = Chem.MolFromSmiles("Cc1ccco1")
    thf = Chem.MolFromSmiles("C1CCOC1")
    morph_f = Chem.MolFromSmiles("C1COCCN1")
    assert mfur is not None and thf is not None and morph_f is not None
    assert count_furan(mfur) >= 1
    assert count_furan(thf) == 0 and count_furan(morph_f) == 0
    assert substructure_alert_excesses(mfur)["furan"] >= 1.0
    assert substructure_alert_excesses(thf)["furan"] == 0.0
    for smi in ("CC#CC", "C#Cc1ccccc1", "CCCC", "Cc1ccco1", "C1CCOC1", "NCCO"):
        b = scorer_z.compute_batch([smi])
        assert b.gate_ok[0] is True and math.isfinite(b.policy[0])

    # Azacrown / vicinal diamine-diol bonuses; Cl/Br/I, sulfonyl, salts ungated.
    from objectives.co2_reward import W_AZACROWN_BONUS, W_VICINAL_DIOL_BONUS

    assert W_AZACROWN_BONUS == 1.0 and W_VICINAL_DIOL_BONUS == 2.0
    pip = Chem.MolFromSmiles("C1CNCCN1")
    tacn = Chem.MolFromSmiles("C1CNCCNCCN1")
    cyclen = Chem.MolFromSmiles("C1CNCCNCCNCCN1")
    assert pip is not None and tacn is not None and cyclen is not None
    assert max_ring_aliphatic_n(pip) == 2
    assert azacrown_bonus_units(pip) == 0.0
    assert max_ring_aliphatic_n(tacn) == 3
    assert azacrown_bonus_units(tacn) == 1.0
    assert max_ring_aliphatic_n(cyclen) == 4
    assert azacrown_bonus_units(cyclen) == 2.0
    hexa = Chem.MolFromSmiles("C1CNCCNCCNCCNCCNCCN1")
    assert hexa is not None
    assert azacrown_bonus_units(hexa) == 2.0
    eda = Chem.MolFromSmiles("NCCN")
    mea = Chem.MolFromSmiles("NCCO")
    vic = Chem.MolFromSmiles("NC(O)C(O)N")
    vic_ch2oh = Chem.MolFromSmiles("NC(CO)C(N)CO")
    assert eda is not None and mea is not None and vic is not None
    assert vicinal_diamine_diol(eda) == 0.0
    assert vicinal_diamine_diol(mea) == 0.0
    assert vicinal_diamine_diol(vic) == 1.0
    assert vicinal_diamine_diol(vic_ch2oh) == 1.0
    assert co2_gate_ok(Chem.MolFromSmiles("CCSCC")) is True
    assert co2_gate_ok(Chem.MolFromSmiles("CCBr")) is False
    assert co2_gate_ok(Chem.MolFromSmiles("CCCl")) is False
    assert co2_gate_ok(Chem.MolFromSmiles("CCI")) is False
    assert count_sulfonyl(Chem.MolFromSmiles("CS(=O)(=O)C")) >= 1
    assert co2_gate_ok(Chem.MolFromSmiles("CS(=O)(=O)C")) is False
    salt = Chem.MolFromSmiles("CCO.CCN")
    assert salt is not None and n_fragments(salt) == 2
    assert co2_gate_ok(salt) is False
    for smi in ("CCBr", "CS(=O)(=O)C", "CCO.CCN"):
        b = scorer_z.compute_batch([smi])
        assert b.gate_ok[0] is False
    thio_b = scorer_z.compute_batch(["CCSCC"])
    assert thio_b.gate_ok[0] is True
    tacn_b = scorer_z.compute_batch(["C1CNCCNCCN1"])
    assert tacn_b.gate_ok[0] is True and math.isfinite(tacn_b.policy[0])
    # Vicinal N–C–C–N (graph dist 3) is skipped by NN cut=4 even if close in space.
    vic_nn = Chem.AddHs(Chem.MolFromSmiles("NCCN"))
    vconf = Chem.Conformer(vic_nn.GetNumAtoms())
    vn = [a.GetIdx() for a in vic_nn.GetAtoms() if a.GetAtomicNum() == 7]
    assert len(vn) == 2
    vconf.SetAtomPosition(vn[0], (0.0, 0.0, 0.0))
    vconf.SetAtomPosition(vn[1], (2.0, 0.0, 0.0))
    for i in range(vic_nn.GetNumAtoms()):
        if i not in vn:
            vconf.SetAtomPosition(i, (0.5, 0.5, 0.5))
    vic_nn.AddConformer(vconf, assignId=True)
    d_vic = min_nonbonded_nn_distance(vic_nn, vic_nn.GetConformer())
    assert math.isnan(d_vic)
    assert nn_distance_excess(d_vic) == 0.0

    # Per-atom physchem: same polar head, extra bulk must not raise physchem.
    from objectives.co2_reward import score_co2_local, _make_mordred_calculator

    local = score_co2_local(
        DIAMINE,
        _make_mordred_calculator(),
        None,
        include_pfc=False,
    )
    assert local.gate_ok
    assert abs(local.hae_mult - 1.0) < 1e-12
    assert local.n_ha > 0
    assert math.isfinite(local.hae_raw) and math.isfinite(local.physchem_z)

    mea_local = score_co2_local(
        "NCCO", _make_mordred_calculator(), None, include_pfc=False
    )
    bulky_local = score_co2_local(
        "CCCCCCCCCCCCNCCO",
        _make_mordred_calculator(),
        None,
        include_pfc=False,
    )
    assert mea_local.gate_ok and bulky_local.gate_ok
    assert bulky_local.n_ha > mea_local.n_ha
    assert mea_local.physchem_z >= bulky_local.physchem_z

    restore_g, streak_g = gate_frac_streak_should_restore(
        0.05, 24, floor=0.20, patience=25
    )
    assert restore_g is True and streak_g == 25
    restore_g2, streak_g2 = gate_frac_streak_should_restore(
        0.5, 10, floor=0.20, patience=25
    )
    assert restore_g2 is False and streak_g2 == 0

    assert abs(valid_fraction(32, 128) - 0.25) < 1e-12
    n_u, u_frac = unique_gated_fraction(["CCO", "CCO", "CCN"], [0, 1, 2], 128)
    assert n_u == 2 and abs(u_frac - 2 / 128) < 1e-12
    health = batch_health_metrics(
        n_valid=3,
        valid_smiles_list=["CCO", "CCO", "CCN"],
        gated_local_indices=[0, 1, 2],
        batch_size=128,
    )
    assert health["n_unique_gated"] == 2
    assert abs(float(health["valid_frac"]) - 3 / 128) < 1e-12

    # Duplicate canonicals → keep highest-reward index only.
    uniq_idx = unique_canonical_learn_indices(
        ["CCO", "OCC", "CCN"],
        [0, 1, 2],
        np.array([1.0, 3.0, 2.0]),
    )
    assert set(int(i) for i in uniq_idx) == {1, 2}

    from objectives.co2_reward import tanimoto_diversity_scores

    dup = tanimoto_diversity_scores(["CCO", "CCO", "CCO"])
    assert dup[0] == 0.0 and dup[1] == 0.0 and dup[2] == 0.0
    single = tanimoto_diversity_scores(["CCO"])
    assert single[0] == 0.0
    diverse = tanimoto_diversity_scores(["CCO", "CCN", "CCC"])
    assert diverse[0] > 0.5

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert abs(float(np.mean(whiten_advantages(x)))) < 1e-9
    # Baseline-relative: batch above baseline → positive mean advantage
    adv = baseline_relative_advantages(np.array([0.5, 0.6, 0.7]), baseline=0.4, std_ema=0.1)
    assert float(np.mean(adv)) > 0.0

    assert should_skip_thin_batch(7, 8) is True
    assert should_skip_thin_batch(8, 8) is False

    idx = top_quantile_indices(np.array([0.1, 0.9, 0.5, 0.2]), 0.25)
    assert idx.size == 1 and int(idx[0]) == 1
    assert top_quantile_indices(np.array([0.1, 0.9, 0.5, 0.2]), 1.0).size == 4
    # Batch of 48 gated → top 25% learns 12
    assert top_quantile_indices(np.arange(48, dtype=np.float64), 0.25).size == 12

    pad_id = 99
    targets = np.array([[1, 2, 3, pad_id, 4, 5], [1, pad_id, 2, 3, 4, 5]], dtype=np.int64)
    mask = pad_credit_mask(targets, pad_id)
    assert mask.dtype == bool
    assert mask[0].tolist() == [True, True, True, False, False, False]
    assert mask[1].tolist() == [True, False, False, False, False, False]
    logps = np.array([[0.1, 0.2, 0.3, 9.0, 9.0, 9.0], [0.5, 9.0, 9.0, 9.0, 9.0, 9.0]])
    seq = (logps * mask).sum(axis=1)
    assert abs(float(seq[0]) - 0.6) < 1e-12
    assert abs(float(seq[1]) - 0.5) < 1e-12

    restore, streak = decline_streak_should_restore(
        0.1, 0.3, 199, margin=0.08, patience=200
    )
    assert restore is True and streak == 200
    restore2, streak2 = decline_streak_should_restore(
        0.25, 0.3, 10, margin=0.08, patience=200
    )
    assert restore2 is False and streak2 == 0
    # Relative: below best*(1-margin) for patience episodes
    restore_r, streak_r = decline_streak_should_restore(
        0.1, 0.3, 199, margin=0.08, patience=200, relative=True
    )
    assert restore_r is True and streak_r == 200
    # 0.28 > 0.3*0.92=0.276 → not in decline
    restore_r2, streak_r2 = decline_streak_should_restore(
        0.28, 0.3, 10, margin=0.08, patience=200, relative=True
    )
    assert restore_r2 is False and streak_r2 == 0

    ema = update_ema(float("nan"), 0.4, 0.9)
    assert abs(ema - 0.4) < 1e-12

    assert abs(linear_schedule(0.15, 0.05, 0.0) - 0.15) < 1e-12
    assert abs(linear_schedule(0.15, 0.05, 1.0) - 0.05) < 1e-12
    assert abs(linear_schedule(0.0, 1.0, 0.5) - 0.5) < 1e-12
    assert abs(explore_schedule_progress(1, 1000) - 0.0) < 1e-12
    assert abs(explore_schedule_progress(501, 1000) - 0.5) < 1e-12
    assert abs(explore_schedule_progress(1001, 1000) - 1.0) < 1e-12
    assert abs(explore_schedule_progress(5000, 1000) - 1.0) < 1e-12
    # top_q anneal: 1.0 → 0.25 over explore horizon
    assert abs(linear_schedule(1.0, 0.25, 0.0) - 1.0) < 1e-12
    assert abs(linear_schedule(1.0, 0.25, 0.5) - 0.625) < 1e-12
    assert abs(linear_schedule(1.0, 0.25, 1.0) - 0.25) < 1e-12
    assert top_quantile_indices(np.arange(100, dtype=np.float64), 1.0).size == 100
    assert top_quantile_indices(np.arange(100, dtype=np.float64), 0.625).size == 63
    assert early_stop_should_stop(499, 500) is False
    assert early_stop_should_stop(500, 500) is True
    assert early_stop_should_stop(999, 0) is False

    # PFC determinism + cache
    clear_pfc_cache()
    pfc = PolarFocalConcavity(n_conformers=2)
    smi = "c1ccccc1O"
    a = pfc.compute(smi)
    n1 = pfc_cache_size()
    b = pfc.compute(smi)
    n2 = pfc_cache_size()
    assert a["composite"] == b["composite"]
    assert n1 == 1 and n2 == 1

    # Voxel vectorization parity vs pure-Python reference + collect_centers flag
    step = 0.5
    for smi_v in ("CCO", "c1ccccc1O", "CC(=O)O"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi_v))
        params = AllChem.ETKDGv3()
        params.randomSeed = 7
        assert AllChem.EmbedMolecule(mol, params) == 0
        conf = mol.GetConformer()
        positions = PolarFocalConcavity._conformer_positions(mol, conf)
        atoms = list(mol.GetAtoms())
        atomic_nums = np.array([a.GetAtomicNum() for a in atoms], dtype=int)
        ref = _voxel_ratio_ref(positions, atomic_nums, step)
        r0, _, f0, h0 = PolarFocalConcavity._voxel_grid_classification(
            positions, atoms, step, collect_centers=False
        )
        r1, _, f1, h1 = PolarFocalConcavity._voxel_grid_classification(
            positions, atoms, step, collect_centers=True
        )
        assert abs(r0 - ref) < 1e-12, (smi_v, r0, ref)
        assert abs(r1 - ref) < 1e-12, (smi_v, r1, ref)
        assert f0.shape == (0, 3) and h0.shape == (0, 3)
        assert f1.shape[0] + h1.shape[0] > 0 or ref == 0.0

    clear_pfc_cache()
    t0 = time.perf_counter()
    timed = PolarFocalConcavity(n_conformers=2).compute("c1ccccc1O")
    dt = time.perf_counter() - t0
    assert math.isfinite(timed["composite"])
    print(f"PFC smoke timing phenol n_conformers=2: {dt:.3f}s composite={timed['composite']:.4f}")

    _check_serial_vs_parallel_parity()

    print("check_co2_rl_explore: ok")


def _finite_close(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= tol


def _check_serial_vs_parallel_parity() -> None:
    """Fixed-batch serial vs ProcessPool rewards must match (incl. scaffold DF)."""
    smiles = [
        DIAMINE,
        "NCCNCCCCCCCCN",
        "NCCN(CCN)CCN",
        "CN(C)CCN(C)CCN(C)C",
        "NCCN1CCN(CCN)CC1",
        "NCCOCCOCCOCCOCCN",
        "NCCCCCCCCCCCCN",
        "NCCCNCCCNCCCN",
    ]
    common = dict(
        random_seed=42,
        n_conformers=2,
        scaffold_bucket=2,
        scaffold_penalty=0.5,
        use_scaffold_filter=True,
    )

    clear_pfc_cache()
    serial = CO2ObjectiveScorer(**common, n_workers=0)
    try:
        b0 = serial.compute_batch(smiles)
    finally:
        serial.close()

    clear_pfc_cache()
    parallel = CO2ObjectiveScorer(**common, n_workers=2)
    try:
        b1 = parallel.compute_batch(smiles)
    finally:
        parallel.close()

    assert b0.gate_ok == b1.gate_ok, (b0.gate_ok, b1.gate_ok)
    for name, xs, ys in (
        ("policy", b0.policy, b1.policy),
        ("physchem", b0.physchem, b1.physchem),
        ("quality", b0.quality, b1.quality),
        ("mw_scale", b0.mw_scale, b1.mw_scale),
    ):
        for i, (x, y) in enumerate(zip(xs, ys)):
            assert _finite_close(float(x), float(y)), (name, i, x, y, smiles[i])
    print("serial vs parallel reward parity: ok")


if __name__ == "__main__":
    main()
