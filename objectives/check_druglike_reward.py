#!/usr/bin/env python3
"""Assert-based self-check for the druglike multi-objective.

Run: MGW_OBJECTIVE=druglike PYTHONPATH=. uv run python objectives/check_druglike_reward.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("MGW_OBJECTIVE", "druglike")

from objectives.druglike_reward import (
    DruglikeObjectiveScorer,
    INVALID_PENALTY,
    W_DIV,
    W_LOGP,
    W_QED,
    W_SA,
    logp_term,
    murcko_rarity_scores,
    qed_term,
    sa_term,
)
from rdkit import Chem
from reward import compute_reward_batch


def main() -> None:
    assert abs(sa_term(1.0) - (-1.0)) < 1e-12
    assert abs(sa_term(5.0) - (-5.0)) < 1e-12
    assert sa_term(2.0) > sa_term(4.0)  # easier synth ranks higher

    assert abs(qed_term(0.5)) < 1e-12
    assert qed_term(0.9) > qed_term(0.5) > qed_term(0.1)
    assert qed_term(0.999) > qed_term(0.99)  # no [0, 1] ceiling

    assert abs(logp_term(2.5) - 2.5) < 1e-12
    assert logp_term(8.0) > logp_term(2.5) > logp_term(-1.0)

    bz = Chem.MolFromSmiles("c1ccccc1")
    tol = Chem.MolFromSmiles("Cc1ccccc1")  # Murcko = benzene
    pyr = Chem.MolFromSmiles("c1ccncc1")
    # Same scaffold → −log(1)=0; distinct → −log(1/2)=log(2)
    assert abs(murcko_rarity_scores([bz, tol])[0] - 0.0) < 1e-12
    assert abs(murcko_rarity_scores([bz, pyr])[0] - math.log(2.0)) < 1e-12
    mixed_r = murcko_rarity_scores([bz, tol, pyr])
    assert abs(mixed_r[0] - (-math.log(2.0 / 3.0))) < 1e-12
    assert abs(mixed_r[2] - (-math.log(1.0 / 3.0))) < 1e-12
    assert mixed_r[2] > mixed_r[0]

    scorer = DruglikeObjectiveScorer()
    smiles = ["c1ccccc1", "CCO", "not_a_smiles", "c1ccncc1"]
    batch = scorer.compute_batch(smiles)
    assert len(batch.policy) == len(smiles)
    assert len(batch.raw_total) == len(smiles)
    assert len(batch.gate_ok) == len(smiles)

    # Benzene / ethanol / pyridine parse; garbage does not.
    assert batch.gate_ok[0] and batch.gate_ok[1] and batch.gate_ok[3]
    assert not batch.gate_ok[2]
    assert batch.policy[2] == INVALID_PENALTY
    # logit(QED) can be negative for modest QED.
    assert batch.sa_good[0] < 0.0
    # Raw MolLogP for benzene is ~1.7, not a peaked ≤0 term.
    assert batch.logp_soft[0] > 0.0

    # Three distinct keys → −log(1/3)=log(3) each
    expect_div = math.log(3.0)
    assert abs(batch.diversity[0] - expect_div) < 1e-12
    assert abs(batch.diversity[1] - expect_div) < 1e-12
    assert abs(batch.diversity[3] - expect_div) < 1e-12
    assert batch.diversity[2] == INVALID_PENALTY

    expected = (
        W_QED * batch.qed[0]
        + W_SA * batch.sa_good[0]
        + W_LOGP * batch.logp_soft[0]
        + W_DIV * batch.diversity[0]
    )
    assert abs(batch.raw_total[0] - expected) < 1e-9
    assert abs(batch.quality[0] - expected) < 1e-9

    # Shared benzene scaffold vs singleton naphthalene → rarity gradient in policy
    mixed = scorer.compute_batch(
        ["c1ccccc1", "Cc1ccccc1", "Clc1ccccc1", "c1ccc2ccccc2c1"]
    )
    assert abs(mixed.diversity[0] - (-math.log(3.0 / 4.0))) < 1e-12
    assert abs(mixed.diversity[3] - (-math.log(1.0 / 4.0))) < 1e-12
    assert mixed.policy[3] > mixed.policy[0]

    # reward.py wiring
    rb = compute_reward_batch(smiles)
    assert len(rb.policy) == len(smiles)
    assert rb.gate_ok == batch.gate_ok
    # Slot map: raw_ba=logit(QED), raw_physchem=−SCScore, mw_scale=MolLogP
    assert abs(rb.raw_ba[0] - batch.qed[0]) < 1e-12
    assert abs(rb.raw_physchem[0] - batch.sa_good[0]) < 1e-12
    assert abs(rb.mw_scale[0] - batch.logp_soft[0]) < 1e-12

    print("check_druglike_reward: ok")


if __name__ == "__main__":
    main()
