#!/usr/bin/env python3
"""Assert-based self-check for the hierarchical chemical tether.

Run: PYTHONPATH=. uv run python objectives/check_chemical_tether.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from objectives.chemical_tether import (
    ChemicalTether,
    HashStubEncoder,
    compute_channel_scales,
    explore_cost,
    loo_channel_distances,
    morgan_bits,
    save_bank,
    tanimoto_nn_dist,
    tether_cost,
    tether_loss,
    union_manifold_smiles,
)


def main() -> None:
    benzene = "c1ccccc1"
    toluene = "c1ccccc1C"
    ethanol = "CCO"
    methane = "C"

    bits_b = morgan_bits(benzene)
    bits_t = morgan_bits(toluene)
    bits_e = morgan_bits(ethanol)
    assert bits_b is not None and bits_t is not None and bits_e is not None

    # Exact self-match → Tanimoto distance 0.
    d_self = tanimoto_nn_dist(bits_b[None, :], bits_b[None, :], k=1)
    assert abs(float(d_self[0])) < 1e-12, d_self

    bank_fps = np.stack([bits_b, bits_t, bits_e], axis=0)
    stub = HashStubEncoder(dim=16)
    e2, v2 = stub.encode([benzene, toluene, ethanol])
    assert v2.all()
    e3, v3 = stub.encode([benzene, toluene, ethanol])
    assert v3.all()

    tether = ChemicalTether(
        smiles=[benzene, toluene, ethanol],
        morgan=bank_fps,
        emb_2d=e2,
        emb_3d=e3,
        lambda_morgan=1.0,
        lambda_2d=1.0,
        lambda_3d=1.0,
        k=1,
        encoder_2d=stub,
        encoder_3d=stub,
    )
    d_hit, ch_hit = tether.distances([benzene])
    assert float(ch_hit["morgan"][0]) < 1e-12, ch_hit["morgan"]
    assert float(ch_hit["2d"][0]) < 1e-5, ch_hit["2d"]
    assert float(ch_hit["3d"][0]) < 1e-5, ch_hit["3d"]
    assert float(d_hit[0]) < 1e-4, d_hit

    d_far, ch_far = tether.distances([methane])
    assert float(ch_far["morgan"][0]) > 0.2, ch_far["morgan"]
    assert float(d_far[0]) > 0.2, d_far

    # λ=0 skips a channel (NaN logged, not mixed into d).
    tether_m = ChemicalTether(
        smiles=[benzene, toluene, ethanol],
        morgan=bank_fps,
        emb_2d=e2,
        emb_3d=e3,
        lambda_morgan=1.0,
        lambda_2d=0.0,
        lambda_3d=0.0,
        k=1,
        encoder_2d=stub,
        encoder_3d=stub,
    )
    d_m, ch_m = tether_m.distances([benzene])
    assert np.isnan(ch_m["2d"][0]) and np.isnan(ch_m["3d"][0])
    # Self-match raw=0 → calibrated 0, independent of s_morgan.
    assert abs(float(d_m[0]) - 0.0) < 1e-12

    unit = ChemicalTether(
        smiles=[benzene, toluene, ethanol],
        morgan=bank_fps,
        emb_2d=e2,
        emb_3d=e3,
        k=1,
        encoder_2d=stub,
        encoder_3d=stub,
        scales={"morgan": 1.0, "2d": 1.0, "3d": 1.0},
    )
    # Identity scales: raw weighted mean (0.6+0.3+0.0)/3 = 0.3.
    d_avg = unit.combined_distance(
        {
            "morgan": np.asarray([0.6], dtype=np.float64),
            "2d": np.asarray([0.3], dtype=np.float64),
            "3d": np.asarray([0.0], dtype=np.float64),
        },
        1,
    )
    assert abs(float(d_avg[0]) - 0.3) < 1e-12, d_avg

    # A: d' = d / s; typical raw distances → combined ≈ 1.
    scaled = ChemicalTether(
        smiles=[benzene, toluene, ethanol],
        morgan=bank_fps,
        emb_2d=e2,
        emb_3d=e3,
        k=1,
        encoder_2d=stub,
        encoder_3d=stub,
        scales={"morgan": 0.6, "2d": 0.08, "3d": 0.008},
    )
    d_typ = scaled.combined_distance(
        {
            "morgan": np.asarray([0.6], dtype=np.float64),
            "2d": np.asarray([0.08], dtype=np.float64),
            "3d": np.asarray([0.008], dtype=np.float64),
        },
        1,
    )
    assert abs(float(d_typ[0]) - 1.0) < 1e-12, d_typ
    primed = scaled.calibrate_channels(
        {
            "morgan": np.asarray([0.6], dtype=np.float64),
            "2d": np.asarray([0.08], dtype=np.float64),
            "3d": np.asarray([0.008], dtype=np.float64),
        }
    )
    assert abs(float(primed["morgan"][0]) - 1.0) < 1e-12
    assert abs(float(primed["2d"][0]) - 1.0) < 1e-12
    assert abs(float(primed["3d"][0]) - 1.0) < 1e-12
    # Failed 2D dropped and λ renormalized: (1 + 1) / 2 = 1.
    d_drop = scaled.combined_distance(
        {
            "morgan": np.asarray([0.6], dtype=np.float64),
            "2d": np.asarray([np.nan], dtype=np.float64),
            "3d": np.asarray([0.008], dtype=np.float64),
        },
        1,
    )
    assert abs(float(d_drop[0]) - 1.0) < 1e-12, d_drop
    # λ still votes after calibration: λ_m=2, others 1, all d'=1 → 1.
    scaled.lambda_morgan = 2.0
    d_vote = scaled.combined_distance(
        {
            "morgan": np.asarray([0.6], dtype=np.float64),
            "2d": np.asarray([0.08], dtype=np.float64),
            "3d": np.asarray([0.008], dtype=np.float64),
        },
        1,
    )
    assert abs(float(d_vote[0]) - 1.0) < 1e-12, d_vote
    scaled.lambda_morgan = 1.0

    # Leave-one-out scales are positive and not ~1 for distinct mols.
    s = compute_channel_scales(bank_fps, e2, e3, k=1)
    assert s["morgan"] > 0.05, s
    assert abs(s["morgan"] - 1.0) > 1e-6, s
    loo = loo_channel_distances(bank_fps, e2, e3, k=1)
    assert loo["morgan"].shape == (3,)
    assert float(np.median(loo["morgan"])) == s["morgan"]
    # Self is excluded: LOO distance is not ~0 for three distinct mols.
    assert float(np.min(loo["morgan"])) > 1e-6

    # REINFORCE tether_loss is not a detached no-op: high-d sample gets +grad on logπ.
    lp = torch.tensor([-1.0, -2.0], dtype=torch.float32, requires_grad=True)
    dist = torch.tensor([0.1, 0.9], dtype=torch.float32)
    loss = tether_loss(lp, dist)  # baseline=None → batch mean
    loss.backward()
    assert lp.grad is not None
    # centered d = [-0.4, 0.4]; grad = centered / n
    assert float(lp.grad[1]) > 0.0, lp.grad
    assert float(lp.grad[0]) < 0.0, lp.grad

    # EMA-style fixed baseline: whole-batch high cost still yields nonzero loss/grad.
    lp2 = torch.tensor([-1.0, -1.0], dtype=torch.float32, requires_grad=True)
    high = torch.tensor([0.9, 0.9], dtype=torch.float32)
    loss_batch = tether_loss(lp2, high)  # centered → 0
    assert abs(float(loss_batch)) < 1e-8
    loss_ema = tether_loss(lp2, high, baseline=0.1)
    assert float(loss_ema) != 0.0
    loss_ema.backward()
    assert lp2.grad is not None
    # cost>baseline → positive ∂L/∂logπ → minimizing L decreases logπ
    assert float(lp2.grad[0]) > 0.0, lp2.grad

    # n=1 → zero loss without baseline; with baseline still defined.
    z = tether_loss(torch.tensor([-1.0]), torch.tensor([0.5]))
    assert float(z) == 0.0
    z_b = tether_loss(torch.tensor([-1.0]), torch.tensor([0.5]), baseline=0.0)
    assert abs(float(z_b) - (-1.0 * 0.5)) < 1e-6
    # NaN sample dropped; one finite → zero loss without baseline.
    z_nan = tether_loss(
        torch.tensor([-1.0, -2.0]), torch.tensor([0.5, float("nan")])
    )
    assert float(z_nan) == 0.0

    # Mode I / Mode II cost mapping.
    d_vec = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float32)
    reg = tether_cost(d_vec, mode="regularize", d_star=0.3)
    assert torch.allclose(reg, d_vec)
    at_star = explore_cost(torch.tensor([0.3]), 0.3)
    assert abs(float(at_star[0])) < 1e-12
    away = explore_cost(torch.tensor([0.1, 0.5]), 0.3)
    assert float(away[0]) > 0.0 and float(away[1]) > 0.0
    shell = tether_cost(d_vec, mode="explore", d_star=0.3)
    assert torch.allclose(shell, (d_vec - 0.3) ** 2)
    try:
        tether_cost(d_vec, mode="nope")
        raise AssertionError("expected ValueError for unknown mode")
    except ValueError:
        pass

    # Union manifold dedupes canonical SMILES.
    union = union_manifold_smiles(
        ["c1ccccc1", "CCO"],
        ["C1=CC=CC=C1", "c1ccccc1C"],
    )
    assert union[0] == "c1ccccc1"
    assert "CCO" in union
    assert len(union) == 3

    # Round-trip npz (morgan-only bank).
    tmp = Path("/tmp/mgw_chem_tether_check.npz")
    save_bank(tmp, [benzene, toluene, ethanol], bank_fps, emb_2d=e2, emb_3d=e3)
    loaded = ChemicalTether.from_npz(
        tmp,
        lambda_morgan=1.0,
        lambda_2d=1.0,
        lambda_3d=1.0,
        encoder_2d=stub,
        encoder_3d=stub,
    )
    d2, _ = loaded.distances([benzene])
    assert float(d2[0]) < 1e-4
    dumped = np.load(tmp, allow_pickle=True)
    assert "scale_morgan" in dumped.files
    assert loaded.scales["morgan"] > 0.05

    save_bank(tmp, [benzene, toluene, ethanol], bank_fps)
    morgan_only = ChemicalTether.from_npz(
        tmp,
        lambda_morgan=1.0,
        lambda_2d=0.0,
        lambda_3d=0.0,
    )
    d3, ch3 = morgan_only.distances([benzene])
    assert float(ch3["morgan"][0]) < 1e-12
    assert np.isnan(ch3["2d"][0])
    tmp.unlink(missing_ok=True)

    print("chemical_tether self-check: ok")


if __name__ == "__main__":
    main()
