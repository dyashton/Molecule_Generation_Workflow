#!/usr/bin/env python3
"""Assert CO2 local scoring speedup from REWARD_WORKERS on unique molecules.

Run: PYTHONPATH=. uv run python objectives/bench_reward_workers.py
"""
from __future__ import annotations

import time

from rdkit import Chem

from objectives.co2_polar_focal_concavity import clear_pfc_cache
from objectives.co2_reward import CO2ObjectiveScorer


def _unique_smiles(n: int = 96) -> list[str]:
    """Build n distinct, parseable SMILES (PFC cache must not dominate the bench)."""
    out: list[str] = []
    seen: set[str] = set()
    i = 0
    while len(out) < n:
        i += 1
        bits: list[str] = []
        x = i
        while x:
            bits.append("C" if x & 1 else "N")
            x >>= 1
            if len(bits) > 14:
                break
        core = "".join(bits) or "C"
        smi = f"c1ccccc1{core}O"
        if smi in seen:
            smi = f"c1ccccc1{core}{'C' * (len(out) + 1)}O"
        if smi in seen:
            continue
        if Chem.MolFromSmiles(smi) is None:
            continue
        seen.add(smi)
        out.append(smi)
    return out


def _time_batch(smiles: list[str], *, n_workers: int) -> float:
    clear_pfc_cache()
    scorer = CO2ObjectiveScorer(
        random_seed=42,
        n_conformers=2,
        use_scaffold_filter=False,
        n_workers=n_workers,
    )
    if n_workers == 0:
        scorer.pfc.embed_num_threads = 1
    try:
        # Start workers; do not score the timed set (keeps PFC caches cold).
        scorer.compute_batch(["CCO"])
        clear_pfc_cache()
        t0 = time.perf_counter()
        scorer.compute_batch(smiles)
        return time.perf_counter() - t0
    finally:
        scorer.close()


def main() -> None:
    smiles = _unique_smiles(96)
    assert len(smiles) == len(set(smiles))
    print(f"bench n={len(smiles)} unique molecules")

    t_serial = _time_batch(smiles, n_workers=0)
    print(f"serial (embed_threads=1): {t_serial:.3f}s")

    t_par = _time_batch(smiles, n_workers=4)
    speedup = t_serial / t_par if t_par > 0 else 0.0
    print(f"parallel workers=4:       {t_par:.3f}s  speedup={speedup:.2f}x")

    assert speedup >= 1.5, (
        f"expected >=1.5x speedup from 4 workers, got {speedup:.2f}x "
        f"(serial={t_serial:.3f}s parallel={t_par:.3f}s)"
    )
    print("reward worker speedup: ok")


if __name__ == "__main__":
    main()
