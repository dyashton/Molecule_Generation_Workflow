"""Assert-based self-check for unbounded proxy objectives.

Run: MGW_OBJECTIVE=unbounded_proxy MGW_PROXY=mollogp \\
       PYTHONPATH=. uv run python objectives/check_unbounded_proxy_reward.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MGW_OBJECTIVE", "unbounded_proxy")
os.environ.setdefault("MGW_PROXY", "mollogp")

from objectives.unbounded_proxy_reward import (
    PROXY_NAMES,
    UnboundedProxyScorer,
    mw_oversize_penalty,
    normalize_proxy_name,
    proxy_from_mol,
)
from rdkit import Chem
from rdkit.Chem import Descriptors
from reward import compute_reward_batch


def main() -> None:
    assert normalize_proxy_name("logp") == "mollogp"
    assert normalize_proxy_name("MW") == "molwt"
    for name in PROXY_NAMES:
        assert normalize_proxy_name(name) == name

    assert abs(mw_oversize_penalty(350.0)) < 1e-12
    assert abs(mw_oversize_penalty(400.0)) < 1e-12
    assert mw_oversize_penalty(450.0, start=400.0, tau=50.0) > 1.0
    assert mw_oversize_penalty(500.0, start=400.0, tau=50.0) > mw_oversize_penalty(
        450.0, start=400.0, tau=50.0
    )
    assert abs(mw_oversize_penalty(900.0, start=0.0)) < 1e-12  # disabled

    mol = Chem.MolFromSmiles("CCO")  # ethanol
    assert abs(proxy_from_mol(mol, "mollogp") - Descriptors.MolLogP(mol)) < 1e-9
    assert abs(proxy_from_mol(mol, "molwt") - Descriptors.MolWt(mol)) < 1e-9
    assert abs(proxy_from_mol(mol, "heavyatomcount") - 3.0) < 1e-9
    assert proxy_from_mol(mol, "bertzct") > 0
    assert proxy_from_mol(mol, "numrotatablebonds") >= 0
    assert proxy_from_mol(mol, "tpsa") > 0
    assert abs(
        proxy_from_mol(mol, "mollogp_molwt")
        - (Descriptors.MolLogP(mol) + Descriptors.MolWt(mol))
    ) < 1e-9

    # Hexane should outrank ethanol on MolLogP (both well under MW start).
    os.environ["MGW_PROXY"] = "mollogp"
    os.environ["MGW_MW_PENALTY_START"] = "400"
    os.environ["MGW_MW_PENALTY_TAU"] = "50"
    batch = UnboundedProxyScorer().compute_batch(["CCO", "CCCCCC", "not_a_smiles"])
    assert batch.gate_ok == [True, True, False]
    assert batch.raw_total[1] > batch.raw_total[0]
    assert batch.policy[1] > batch.policy[0]
    assert batch.policy[2] == -1.0

    # Long alkane: high MolLogP but large MW penalty → score below hexane.
    long_chain = "C" * 40  # C40 ~ 562 Da
    batch_pen = UnboundedProxyScorer("mollogp").compute_batch(["CCCCCC", long_chain])
    assert batch_pen.molwt[1] > 400.0
    assert batch_pen.mollogp[1] > batch_pen.mollogp[0]
    assert batch_pen.raw_total[1] < batch_pen.raw_total[0]

    # MolWt: hexane heavier than ethanol.
    os.environ["MGW_PROXY"] = "molwt"
    batch_w = UnboundedProxyScorer("molwt").compute_batch(["CCO", "CCCCCC"])
    assert batch_w.raw_total[1] > batch_w.raw_total[0]

    # reward.py wiring
    os.environ["MGW_OBJECTIVE"] = "unbounded_proxy"
    os.environ["MGW_PROXY"] = "mollogp"
    rb = compute_reward_batch(["CCO", "CCCCCC"])
    assert rb.raw_total[1] > rb.raw_total[0]
    assert rb.gate_ok == [True, True]
    assert rb.raw_physchem[1] > rb.raw_physchem[0]  # MolWt companion
    assert rb.mw_scale[0] == 3.0  # ethanol HeavyAtomCount
    assert rb.raw_div[0] > 0  # BertzCT

    print("check_unbounded_proxy_reward: ok")


if __name__ == "__main__":
    main()
