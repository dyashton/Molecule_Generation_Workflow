from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

INVALID_DIVERSITY = 0.0


def tanimoto_diversity_scores(
    smiles_list: list[str], *, radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """
    Per-molecule diversity in [0, 1]: mean(1 - Tanimoto) vs other parseable structures.

    This keeps the workflow repo's existing Morgan-fingerprint diversity style.
    """
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
            DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in valid_idx if j != i
        ]
        scores[i] = float(np.mean([1.0 - s for s in sims])) * uniq_scale
    return scores

