from __future__ import annotations

"""Small helpers for CO2 REINFORCE explore / stability (testable without torch)."""

import math

import numpy as np


def should_skip_thin_batch(n_gated: int, min_gated: int) -> bool:
    return int(n_gated) < int(min_gated)


def top_quantile_indices(rewards: np.ndarray, top_quantile: float) -> np.ndarray:
    """Indices of the highest `top_quantile` fraction (at least 1)."""
    n = int(rewards.size)
    if n == 0:
        return np.array([], dtype=np.int64)
    q = float(top_quantile)
    q = min(max(q, 1e-9), 1.0)
    k = max(1, int(math.ceil(n * q)))
    # Highest rewards first
    order = np.argsort(-rewards)
    return order[:k].astype(np.int64)


def whiten_advantages(rewards: np.ndarray) -> np.ndarray:
    """Zero-mean unit-std advantages; empty / constant → zeros."""
    x = np.asarray(rewards, dtype=np.float64)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + 1e-8)


def baseline_relative_advantages(
    rewards: np.ndarray, baseline: float, std_ema: float
) -> np.ndarray:
    """(r - baseline) / (std_ema + eps); preserves positive mean when batch > baseline."""
    x = np.asarray(rewards, dtype=np.float64)
    if x.size == 0:
        return x
    scale = float(std_ema) if math.isfinite(std_ema) and std_ema > 0 else 1.0
    return (x - float(baseline)) / (scale + 1e-8)


def update_ema(prev: float, value: float, momentum: float) -> float:
    """EMA update; if prev is non-finite, return value."""
    m = float(momentum)
    if not math.isfinite(prev):
        return float(value)
    if not math.isfinite(value):
        return float(prev)
    return m * float(prev) + (1.0 - m) * float(value)


def decline_streak_should_restore(
    monitor: float,
    best_reward: float,
    streak: int,
    *,
    margin: float,
    patience: int,
    relative: bool = False,
) -> tuple[bool, int]:
    """
    Update decline streak; return (should_restore, new_streak).

    Absolute (default): restore when monitor < best - margin for `patience` episodes.
    Relative: restore when monitor < best * (1 - margin) for `patience` episodes.
    """
    if not math.isfinite(monitor) or not math.isfinite(best_reward):
        return False, 0
    if relative:
        threshold = float(best_reward) * (1.0 - float(margin))
    else:
        threshold = float(best_reward) - float(margin)
    if monitor < threshold:
        new_streak = int(streak) + 1
    else:
        new_streak = 0
    return new_streak >= int(patience), new_streak


def gate_frac_streak_should_restore(
    gated_frac: float,
    streak: int,
    *,
    floor: float,
    patience: int,
) -> tuple[bool, int]:
    """
    Restore when gated/valid fraction stays below `floor` for `patience` episodes.
    patience<=0 disables.
    """
    if int(patience) <= 0:
        return False, 0
    if not math.isfinite(gated_frac):
        return False, int(streak)
    if float(gated_frac) < float(floor):
        new_streak = int(streak) + 1
    else:
        new_streak = 0
    return new_streak >= int(patience), new_streak


def valid_fraction(n_valid: int, batch_size: int) -> float:
    """Fraction of the generation batch that RDKit parses."""
    if int(batch_size) <= 0:
        return 0.0
    return float(max(int(n_valid), 0)) / float(batch_size)


def unique_gated_fraction(
    valid_smiles_list: list[str],
    gated_local_indices: np.ndarray | list[int],
    batch_size: int,
) -> tuple[int, float]:
    """Unique canonical gated SMILES and their fraction of the full batch."""
    from rdkit import Chem

    unique: set[str] = set()
    for i in gated_local_indices:
        mol = Chem.MolFromSmiles(valid_smiles_list[int(i)])
        if mol is not None:
            unique.add(Chem.MolToSmiles(mol))
    n_unique = len(unique)
    frac = float(n_unique) / float(batch_size) if int(batch_size) > 0 else 0.0
    return n_unique, frac


def batch_health_metrics(
    *,
    n_valid: int,
    valid_smiles_list: list[str],
    gated_local_indices: np.ndarray | list[int],
    batch_size: int,
) -> dict[str, float | int]:
    n_unique, uniq_frac = unique_gated_fraction(
        valid_smiles_list, gated_local_indices, batch_size
    )
    return {
        "valid_frac": valid_fraction(n_valid, batch_size),
        "unique_gated_frac": uniq_frac,
        "n_unique_gated": n_unique,
    }


def unique_canonical_learn_indices(
    valid_smiles_list: list[str],
    candidate_local_indices: np.ndarray | list[int],
    rewards: np.ndarray,
) -> np.ndarray:
    """One local index per canonical SMILES; highest reward wins."""
    from rdkit import Chem

    best: dict[str, tuple[float, int]] = {}
    for i in candidate_local_indices:
        idx = int(i)
        smi = valid_smiles_list[idx]
        mol = Chem.MolFromSmiles(smi)
        can = Chem.MolToSmiles(mol) if mol is not None else smi
        r = float(rewards[idx])
        if not math.isfinite(r):
            continue
        prev = best.get(can)
        if prev is None or r > prev[0]:
            best[can] = (r, idx)
    if not best:
        return np.array([], dtype=np.int64)
    return np.asarray([idx for _r, idx in best.values()], dtype=np.int64)


def canonical_smiles_or_raw(smi: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi


def mw_scale(
    mw: float,
    *,
    lo: float = 50.0,
    hi: float = 400.0,
    inner_lo: float = 100.0,
    inner_hi: float = 350.0,
) -> float:
    """
    Multiplicative MW trapezoid in [0, 1]:
      0 outside (lo, hi], 1 on [inner_lo, inner_hi], linear ramps at the edges.
    """
    x = float(mw)
    if not math.isfinite(x) or x <= lo or x > hi:
        return 0.0
    ilo = max(float(inner_lo), lo + 1e-6)
    ihi = min(float(inner_hi), hi - 1e-6)
    if ilo > ihi:
        ilo, ihi = ihi, ilo
    if ilo <= x <= ihi:
        return 1.0
    if x < ilo:
        return (x - lo) / (ilo - lo)
    return (hi - x) / (hi - ihi)


def high_end_soft_scale(
    x: float,
    *,
    full_hi: float,
    zero_hi: float,
) -> float:
    """
    One-sided multiplicative soft scale in [0, 1]:
      1 for x <= full_hi, linear down to 0 at zero_hi, 0 above.
    """
    v = float(x)
    if not math.isfinite(v):
        return 0.0
    fh = float(full_hi)
    zh = float(zero_hi)
    if zh <= fh:
        return 1.0 if v <= fh else 0.0
    if v <= fh:
        return 1.0
    if v >= zh:
        return 0.0
    return (zh - v) / (zh - fh)


def sa_scale(
    score: float,
    *,
    full_hi: float = 3.5,
    zero_hi: float = 5.0,
) -> float:
    """SCScore soft scale: full credit until full_hi, ramp to 0 at zero_hi."""
    return high_end_soft_scale(score, full_hi=full_hi, zero_hi=zero_hi)


def consec_rot_scale(
    n: float,
    *,
    full_hi: float = 4.0,
    zero_hi: float = 7.0,
) -> float:
    """Max-consecutive-rotatable-bonds soft scale."""
    return high_end_soft_scale(n, full_hi=full_hi, zero_hi=zero_hi)


def count_rotatable_bonds(mol) -> int:
    """Lipinski rotatable-bond count (RDKit Descriptors.NumRotatableBonds)."""
    if mol is None:
        return 0
    from rdkit.Chem import Descriptors

    return int(Descriptors.NumRotatableBonds(mol))


def rot_window_excess(
    n: float,
    *,
    lo: float = 3.0,
    hi: float = 6.0,
) -> float:
    """
    Soft total-rotatable-bond window excess:
      0 on [lo, hi], max(0, lo - n) below, max(0, n - hi) above.
    Examples (lo=3, hi=6): n=2→1, n=1→2, n=7→1, n=8→2.
    """
    v = float(n)
    if not math.isfinite(v):
        return 0.0
    return max(0.0, float(lo) - v) + max(0.0, v - float(hi))


def oh_scale(
    n: float,
    *,
    full_hi: float = 3.0,
    zero_hi: float = 5.0,
) -> float:
    """
    Hydroxyl-count soft scale: full credit ≤ full_hi, linear to 0 at zero_hi.
    Default: ≤3 OH → 1, 4 OH → 0.5, ≥5 OH → 0 (saturating; prefer high_end_excess).
    """
    return high_end_soft_scale(n, full_hi=full_hi, zero_hi=zero_hi)


def excess_from_scale(scale: float) -> float:
    """Unit excess in [0, 1]: 0 when scale=1 (in limits), 1 when scale=0 (saturates)."""
    s = float(scale)
    if not math.isfinite(s):
        return 1.0
    return min(1.0, max(0.0, 1.0 - s))


def high_end_excess(
    x: float,
    *,
    full_hi: float,
    zero_hi: float,
) -> float:
    """
    Uncapped one-sided excess for additive penalties:
      0 for x <= full_hi, 1 at zero_hi, continues linearly past zero_hi.
    """
    v = float(x)
    if not math.isfinite(v):
        return 0.0
    fh = float(full_hi)
    zh = float(zero_hi)
    span = zh - fh
    if span <= 0.0:
        return 0.0 if v <= fh else float("inf")
    return max(0.0, (v - fh) / span)


def mw_excess(
    mw: float,
    *,
    lo: float = 50.0,
    hi: float = 400.0,
    inner_lo: float = 100.0,
    inner_hi: float = 350.0,
) -> float:
    """
    Uncapped MW excess for additive penalties:
      0 on [inner_lo, inner_hi], 1 at outer lo/hi, continues linearly outside.
    """
    x = float(mw)
    if not math.isfinite(x):
        return 0.0
    ilo = max(float(inner_lo), float(lo) + 1e-6)
    ihi = min(float(inner_hi), float(hi) - 1e-6)
    if ilo > ihi:
        ilo, ihi = ihi, ilo
    if ilo <= x <= ihi:
        return 0.0
    if x < ilo:
        denom = ilo - float(lo)
        return (ilo - x) / denom if denom > 0.0 else float("inf")
    denom = float(hi) - ihi
    return (x - ihi) / denom if denom > 0.0 else float("inf")


# RDKit rotatable-bond SMARTS (same class as CalcNumRotatableBonds / Lipinski).
_ROTATABLE_BOND_SMARTS = None
_OH_SMARTS = None
_AMINE_N_SMARTS = None


def _rotatable_bond_smarts():
    global _ROTATABLE_BOND_SMARTS
    if _ROTATABLE_BOND_SMARTS is None:
        from rdkit import Chem

        _ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts(
            "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
        )
    return _ROTATABLE_BOND_SMARTS


def _oh_smarts():
    global _OH_SMARTS
    if _OH_SMARTS is None:
        from rdkit import Chem

        # Neutral hydroxyl (alcohol / phenol); excludes charged [O-] etc.
        _OH_SMARTS = Chem.MolFromSmarts("[OX2H1]")
    return _OH_SMARTS


def _amine_n_smarts():
    global _AMINE_N_SMARTS
    if _AMINE_N_SMARTS is None:
        from rdkit import Chem

        # Aliphatic primary / secondary / tertiary amines (excludes amides, aromatic N).
        _AMINE_N_SMARTS = Chem.MolFromSmarts(
            "[$([NX3H2][CX4]),$([NX3H]([CX4])[CX4]),$([NX3]([CX4])([CX4])[CX4])]"
        )
    return _AMINE_N_SMARTS


def count_oh(mol) -> int:
    """Number of hydroxyl groups (alcohols + phenols). 0 if mol is None."""
    if mol is None:
        return 0
    smarts = _oh_smarts()
    if smarts is None:
        return 0
    return int(len(mol.GetSubstructMatches(smarts)))


def count_amine_n(mol) -> int:
    """Number of aliphatic amine nitrogens (1°/2°/3°). 0 if mol is None."""
    if mol is None:
        return 0
    smarts = _amine_n_smarts()
    if smarts is None:
        return 0
    return int(len(mol.GetSubstructMatches(smarts)))


# RDKit SMARTS caches for soft substructure alerts.
_EPOXIDE_AZIRIDINE_SMARTS = None
_ESTER_SMARTS = None
_CYCLIC_ACETAL_SMARTS = None
_AMINAL_SMARTS = None
_HYDRATE_SMARTS = None
_HEMIAMINAL_SMARTS = None
_PEROXIDE_SMARTS = None
_PERACID_SMARTS = None
_CARBOXYLIC_ACID_SMARTS = None
_AMIDE_SMARTS = None
_HINDERED_AMINE_ALPHA_SMARTS = None
_ALKYNE_SMARTS = None
_FURAN_SMARTS = None
_SULFONYL_SMARTS = None
_VICINAL_PRIMARY_DIAMINE_SMARTS = None


def _epoxide_aziridine_smarts():
    global _EPOXIDE_AZIRIDINE_SMARTS
    if _EPOXIDE_AZIRIDINE_SMARTS is None:
        from rdkit import Chem

        # N or O in a 3-membered ring (epoxide / aziridine heteroatom).
        _EPOXIDE_AZIRIDINE_SMARTS = Chem.MolFromSmarts("[N,O;r3]")
    return _EPOXIDE_AZIRIDINE_SMARTS


def _ester_smarts():
    global _ESTER_SMARTS
    if _ESTER_SMARTS is None:
        from rdkit import Chem

        # Ester (and carbonate-like) C(=O)–O–C; excludes carboxylic acids (OX2H).
        _ESTER_SMARTS = Chem.MolFromSmarts("C(=O)[OX2H0][#6]")
    return _ESTER_SMARTS


def _cyclic_acetal_smarts():
    """Return cached (dioxolane, dioxane) SMARTS tuple."""
    global _CYCLIC_ACETAL_SMARTS
    if not isinstance(_CYCLIC_ACETAL_SMARTS, tuple):
        from rdkit import Chem

        _CYCLIC_ACETAL_SMARTS = (
            Chem.MolFromSmarts("[OX2]1[CX4][OX2][CX4][CX4]1"),  # 1,3-dioxolane
            Chem.MolFromSmarts("[OX2]1[CX4][OX2][CX4][CX4][CX4]1"),  # 1,3-dioxane
        )
    return _CYCLIC_ACETAL_SMARTS


def _aminal_smarts():
    global _AMINAL_SMARTS
    if _AMINAL_SMARTS is None:
        from rdkit import Chem

        # Geminal diamine (aminal): sp3 C with two NX3.
        _AMINAL_SMARTS = Chem.MolFromSmarts("[CX4]([NX3])([NX3])")
    return _AMINAL_SMARTS


def _hydrate_smarts():
    global _HYDRATE_SMARTS
    if _HYDRATE_SMARTS is None:
        from rdkit import Chem

        # Geminal diol (hydrate): sp3 C with two OH.
        _HYDRATE_SMARTS = Chem.MolFromSmarts("[CX4]([OX2H])([OX2H])")
    return _HYDRATE_SMARTS


def _hemiaminal_smarts():
    global _HEMIAMINAL_SMARTS
    if _HEMIAMINAL_SMARTS is None:
        from rdkit import Chem

        # Geminal N+O (hemiaminal / N,O-acetal carbon).
        _HEMIAMINAL_SMARTS = Chem.MolFromSmarts("[CX4]([NX3])([OX2])")
    return _HEMIAMINAL_SMARTS


def _peroxide_smarts():
    global _PEROXIDE_SMARTS
    if _PEROXIDE_SMARTS is None:
        from rdkit import Chem

        # Any O–O (hydroperoxide, dialkyl peroxide, peracid O–O).
        _PEROXIDE_SMARTS = Chem.MolFromSmarts("[OX2][OX2]")
    return _PEROXIDE_SMARTS


def _peracid_smarts():
    global _PERACID_SMARTS
    if _PERACID_SMARTS is None:
        from rdkit import Chem

        # Peracid / perester: C(=O)–O–O.
        _PERACID_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2][OX2]")
    return _PERACID_SMARTS


def _carboxylic_acid_smarts():
    global _CARBOXYLIC_ACID_SMARTS
    if _CARBOXYLIC_ACID_SMARTS is None:
        from rdkit import Chem

        # –COOH only (not esters, not peracids).
        _CARBOXYLIC_ACID_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    return _CARBOXYLIC_ACID_SMARTS


def _amide_smarts():
    global _AMIDE_SMARTS
    if _AMIDE_SMARTS is None:
        from rdkit import Chem

        # Carboxamide / lactam N (urea matches twice).
        _AMIDE_SMARTS = Chem.MolFromSmarts("[NX3][CX3](=O)")
    return _AMIDE_SMARTS


def _hindered_amine_alpha_smarts():
    global _HINDERED_AMINE_ALPHA_SMARTS
    if _HINDERED_AMINE_ALPHA_SMARTS is None:
        from rdkit import Chem

        # Non-amide NX3 on a secondary/tertiary/quaternary carbon (iPr, tBu, cyclohexyl).
        # RDKit rejects [C=O] inside this recursive; NC=O is the same substructure.
        # ponytail: also hits AMP / N-cyclohexyl; tighten to gem-dimethyl if too broad.
        _HINDERED_AMINE_ALPHA_SMARTS = Chem.MolFromSmarts(
            "[NX3;!$(NC=O)][CX4]([#6])([#6])"
        )
    return _HINDERED_AMINE_ALPHA_SMARTS


def _alkyne_smarts():
    global _ALKYNE_SMARTS
    if _ALKYNE_SMARTS is None:
        from rdkit import Chem

        # sp carbon in a C≡C (terminal or internal). Not nitriles.
        _ALKYNE_SMARTS = Chem.MolFromSmarts("[$([CX2]#C)]")
    return _ALKYNE_SMARTS


def _furan_smarts():
    global _FURAN_SMARTS
    if _FURAN_SMARTS is None:
        from rdkit import Chem

        # Aromatic furan only (not THF).
        _FURAN_SMARTS = Chem.MolFromSmarts("o1cccc1")
    return _FURAN_SMARTS


def _sulfonyl_smarts():
    global _SULFONYL_SMARTS
    if _SULFONYL_SMARTS is None:
        from rdkit import Chem

        _SULFONYL_SMARTS = Chem.MolFromSmarts("[SX4](=O)(=O)")
    return _SULFONYL_SMARTS


def _vicinal_primary_diamine_smarts():
    global _VICINAL_PRIMARY_DIAMINE_SMARTS
    if _VICINAL_PRIMARY_DIAMINE_SMARTS is None:
        from rdkit import Chem

        _VICINAL_PRIMARY_DIAMINE_SMARTS = Chem.MolFromSmarts(
            "[NX3H2][CX4][CX4][NX3H2]"
        )
    return _VICINAL_PRIMARY_DIAMINE_SMARTS


def count_smarts_matches(mol, smarts) -> int:
    """Number of unique substructure matches; 0 if mol/smarts is None."""
    if mol is None or smarts is None:
        return 0
    return int(len(mol.GetSubstructMatches(smarts)))


def count_sulfur(mol) -> int:
    """Number of sulfur atoms (any isotope/charge). 0 if mol is None."""
    if mol is None:
        return 0
    return int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16))


def count_nitrogen(mol) -> int:
    """Number of nitrogen atoms (atomic num 7). 0 if mol is None."""
    if mol is None:
        return 0
    return int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7))


def count_excess(n: float, *, free: float = 0.0) -> float:
    """Soft count excess: max(0, n - free)."""
    if not math.isfinite(float(n)):
        return 0.0
    return max(0.0, float(n) - float(free))


def count_epoxide_aziridine(mol) -> int:
    return count_smarts_matches(mol, _epoxide_aziridine_smarts())


def count_ester(mol) -> int:
    return count_smarts_matches(mol, _ester_smarts())


def count_cyclic_acetal(mol) -> int:
    """Count 1,3-dioxolane + 1,3-dioxane rings."""
    if mol is None:
        return 0
    n = 0
    for smarts in _cyclic_acetal_smarts():
        if smarts is not None:
            n += int(len(mol.GetSubstructMatches(smarts)))
    return n


def count_aminal(mol) -> int:
    return count_smarts_matches(mol, _aminal_smarts())


def count_hydrate(mol) -> int:
    return count_smarts_matches(mol, _hydrate_smarts())


def count_hemiaminal(mol) -> int:
    return count_smarts_matches(mol, _hemiaminal_smarts())


def count_peroxide(mol) -> int:
    return count_smarts_matches(mol, _peroxide_smarts())


def count_peracid(mol) -> int:
    return count_smarts_matches(mol, _peracid_smarts())


def count_carboxylic_acid(mol) -> int:
    return count_smarts_matches(mol, _carboxylic_acid_smarts())


def count_amide(mol) -> int:
    return count_smarts_matches(mol, _amide_smarts())


def count_hindered_amine_alpha(mol) -> int:
    """Count unique N–α-C pairs (tBu would otherwise match 3 ways)."""
    if mol is None:
        return 0
    smarts = _hindered_amine_alpha_smarts()
    if smarts is None:
        return 0
    # ponytail: unique pairs; upgrade if we need per-branch counting.
    pairs = {(m[0], m[1]) for m in mol.GetSubstructMatches(smarts)}
    return int(len(pairs))


def count_ring_oxygen(mol) -> int:
    """Oxygen atoms in any ring. 0 if mol is None."""
    if mol is None:
        return 0
    return int(
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8 and a.IsInRing())
    )


def count_alkyne(mol) -> int:
    return count_smarts_matches(mol, _alkyne_smarts())


def count_furan(mol) -> int:
    return count_smarts_matches(mol, _furan_smarts())


CO2_ALLOWED_ATOMS = frozenset({"C", "H", "N", "O", "F", "S"})


def count_sulfonyl(mol) -> int:
    return count_smarts_matches(mol, _sulfonyl_smarts())


def n_fragments(mol) -> int:
    """Disconnected fragment count. 0 if mol is None."""
    if mol is None:
        return 0
    from rdkit import Chem

    return int(len(Chem.GetMolFrags(mol)))


def co2_gate_ok(mol) -> bool:
    """Charge 0, atoms in {C,H,N,O,F,S}, no sulfonyl, one fragment."""
    if mol is None:
        return False
    from rdkit import Chem

    if Chem.GetFormalCharge(mol) != 0:
        return False
    if not all(a.GetSymbol() in CO2_ALLOWED_ATOMS for a in mol.GetAtoms()):
        return False
    if count_sulfonyl(mol) > 0:
        return False
    if n_fragments(mol) != 1:
        return False
    return True


def max_ring_aliphatic_n(mol) -> int:
    """Max non-aromatic N atoms in any one ring. 0 if mol is None."""
    if mol is None:
        return 0
    ri = mol.GetRingInfo()
    best = 0
    for ring in ri.AtomRings():
        n = sum(
            1
            for i in ring
            if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
            and not mol.GetAtomWithIdx(i).GetIsAromatic()
        )
        if n > best:
            best = n
    return int(best)


def azacrown_bonus_units(mol, *, cap: int = 2) -> float:
    """min(cap, max(0, max_ring_aliphatic_N − 2)). Piperazine 0; TACN 1; cyclen 2."""
    return float(min(int(cap), max(0, max_ring_aliphatic_n(mol) - 2)))


def _atom_is_oh(atom) -> bool:
    return atom.GetAtomicNum() == 8 and int(atom.GetTotalNumHs()) >= 1


def _carbon_has_oh_or_ch2oh(mol, c_idx: int, other_c_idx: int) -> bool:
    c = mol.GetAtomWithIdx(c_idx)
    for nbr in c.GetNeighbors():
        if _atom_is_oh(nbr):
            return True
        if nbr.GetAtomicNum() == 6 and int(nbr.GetIdx()) != int(other_c_idx):
            if any(_atom_is_oh(n2) for n2 in nbr.GetNeighbors()):
                return True
    return False


def vicinal_diamine_diol(mol) -> float:
    """1.0 if a primary 1,2-diamine has OH or CH2OH on both carbons; else 0."""
    if mol is None:
        return 0.0
    smarts = _vicinal_primary_diamine_smarts()
    if smarts is None:
        return 0.0
    for match in mol.GetSubstructMatches(smarts):
        _n1, c1, c2, _n2 = match
        if _carbon_has_oh_or_ch2oh(mol, c1, c2) and _carbon_has_oh_or_ch2oh(
            mol, c2, c1
        ):
            return 1.0
    return 0.0


def nonbasic_n_excess(mol) -> float:
    """Soft excess: nitrogens that are not aliphatic basic amines."""
    if mol is None:
        return 0.0
    return max(0.0, float(count_nitrogen(mol)) - float(count_amine_n(mol)))


def substructure_alert_excesses(
    mol,
    *,
    acetal_free: float = 1.0,
    sulfur_free: float = 0.0,
    ring_o_free: float = 1.0,
) -> dict[str, float]:
    """
    Soft SMARTS / S-count excesses for additive penalties.

    Keys: epoxide_aziridine, sulfur, ester, cyclic_acetal,
    aminal, hydrate, hemiaminal, peroxide, peracid, carboxylic_acid,
    amide, hindered_amine_alpha, ring_oxygen, alkyne, furan.
    """
    return {
        "epoxide_aziridine": count_excess(
            float(count_epoxide_aziridine(mol)), free=0.0
        ),
        "sulfur": count_excess(float(count_sulfur(mol)), free=sulfur_free),
        "ester": count_excess(float(count_ester(mol)), free=0.0),
        "cyclic_acetal": count_excess(
            float(count_cyclic_acetal(mol)), free=acetal_free
        ),
        "aminal": count_excess(float(count_aminal(mol)), free=0.0),
        "hydrate": count_excess(float(count_hydrate(mol)), free=0.0),
        "hemiaminal": count_excess(float(count_hemiaminal(mol)), free=0.0),
        "peroxide": count_excess(float(count_peroxide(mol)), free=0.0),
        "peracid": count_excess(float(count_peracid(mol)), free=0.0),
        "carboxylic_acid": count_excess(float(count_carboxylic_acid(mol)), free=0.0),
        "amide": count_excess(float(count_amide(mol)), free=0.0),
        "hindered_amine_alpha": count_excess(
            float(count_hindered_amine_alpha(mol)), free=0.0
        ),
        "ring_oxygen": count_excess(
            float(count_ring_oxygen(mol)), free=ring_o_free
        ),
        "alkyne": count_excess(float(count_alkyne(mol)), free=0.0),
        "furan": count_excess(float(count_furan(mol)), free=0.0),
    }


def amine_justified_oh_excess(
    n_oh: float,
    n_amine: float,
    *,
    free: float = 0.0,
) -> float:
    """
    Uncapped polyol-style OH excess: OH beyond (n_amine + free).

    With free=0, 1:1 alkanolamines (MEA) stay 0; extra OH grows linearly.
    """
    budget = max(0.0, float(n_amine) + float(free))
    return max(0.0, float(n_oh) - budget)


def amine_floor_excess(n_amine: float, *, floor: float = 3.0) -> float:
    """
    Soft low-end amine floor for additive penalties:
      0 when n_amine >= floor, 1 at 0 amines (excess = (floor - n) / floor).
    """
    n = float(n_amine)
    f = float(floor)
    if not math.isfinite(n) or f <= 0.0:
        return 0.0
    return max(0.0, f - n) / f


def per_heavy_atom(x: float, n_ha: int) -> float:
    """Extensive descriptor → per-heavy-atom (n_ha floored at 1)."""
    n = max(int(n_ha), 1)
    v = float(x)
    if not math.isfinite(v):
        return v
    return v / float(n)


def hae_raw(
    bas_u: float,
    tpsa_u: float,
    n_ha: int,
    *,
    pfc_u: float | None = None,
) -> float:
    """
    Heavy-atom efficiency: (basicity + TopoPSA [+ PFC]) unit scores / n_HA.

    Unit scores are term_score outputs (z or sigmoid). Drop PFC when pfc_u is None.
    """
    n = max(int(n_ha), 1)
    r = float(bas_u) + float(tpsa_u)
    if pfc_u is not None:
        r += float(pfc_u)
    return r / float(n)


def hae_multiplier(
    z_hae: float,
    *,
    alpha: float = 0.35,
    lo: float = 0.25,
    hi: float = 3.0,
) -> float:
    """Positive physchem scale from soft-capped z_HAE: clip(1 + alpha * z, lo, hi)."""
    m = 1.0 + float(alpha) * float(z_hae)
    return float(min(float(hi), max(float(lo), m)))


def min_nonbonded_nn_distance(mol_h, conf, *, min_graph_dist: int = 4) -> float:
    """
    Minimum Euclidean N···N distance among pairs with graph distance >= min_graph_dist.

    `mol_h` should include explicit H if the conformer was built that way; N indices
    are taken from atomic numbers. Returns NaN if fewer than two qualifying N atoms
    or no pair meets the graph-distance cut.
    """
    from rdkit import Chem

    if mol_h is None or conf is None:
        return float("nan")
    n_ids = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 7]
    if len(n_ids) < 2:
        return float("nan")
    best = float("inf")
    for i in range(len(n_ids)):
        for j in range(i + 1, len(n_ids)):
            a, b = int(n_ids[i]), int(n_ids[j])
            try:
                path = Chem.rdmolops.GetShortestPath(mol_h, a, b)
                gdist = len(path) - 1
            except Exception:
                continue
            if gdist < int(min_graph_dist):
                continue
            pa = conf.GetAtomPosition(a)
            pb = conf.GetAtomPosition(b)
            d = float(pa.Distance(pb))
            if d < best:
                best = d
    if not math.isfinite(best) or best == float("inf"):
        return float("nan")
    return float(best)


def min_nonbonded_nn_distance_over_confs(
    mol_h, *, min_graph_dist: int = 4
) -> float:
    """Min over all conformers of min_nonbonded_nn_distance; NaN if none usable."""
    if mol_h is None or mol_h.GetNumConformers() == 0:
        return float("nan")
    best = float("inf")
    for cid in range(mol_h.GetNumConformers()):
        d = min_nonbonded_nn_distance(
            mol_h, mol_h.GetConformer(cid), min_graph_dist=min_graph_dist
        )
        if math.isfinite(d) and d < best:
            best = d
    if not math.isfinite(best) or best == float("inf"):
        return float("nan")
    return float(best)


def nn_distance_excess(d_min: float, *, thresh: float = 2.8) -> float:
    """Soft excess in [0, 1+]: max(0, thresh - d_min) / thresh; 0 if d_min is NaN."""
    if not math.isfinite(float(d_min)):
        return 0.0
    t = max(float(thresh), 1e-9)
    return max(0.0, t - float(d_min)) / t


def max_consecutive_rotatable_bonds(mol) -> int:
    """
    Longest simple path (bond count) in the graph of RDKit rotatable bonds.

    0 if mol is None or has no rotatable bonds. Allows many short hinges while
    flagging a single long contiguous arm.
    """
    if mol is None:
        return 0
    smarts = _rotatable_bond_smarts()
    if smarts is None:
        return 0
    matches = mol.GetSubstructMatches(smarts)
    if not matches:
        return 0
    adj: dict[int, set[int]] = {}
    for a, b in matches:
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))

    best = 0

    def _dfs(u: int, visited: set[int], depth: int) -> None:
        nonlocal best
        if depth > best:
            best = depth
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            _dfs(v, visited, depth + 1)
            visited.remove(v)

    for start in adj:
        _dfs(start, {start}, 0)
    return int(best)


def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolate start→end for progress in [0, 1] (clamped)."""
    t = min(max(float(progress), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * t


def explore_schedule_progress(episode: int, horizon: int) -> float:
    """Linear explore progress: ep 1 → 0, ep horizon+1 → 1 (clamped)."""
    h = max(int(horizon), 1)
    return min(max(float(int(episode) - 1) / float(h), 0.0), 1.0)


def early_stop_should_stop(episodes_since_best: int, patience: int) -> bool:
    """True when no new best for `patience` episodes. patience<=0 disables."""
    if int(patience) <= 0:
        return False
    return int(episodes_since_best) >= int(patience)


def pad_credit_mask(targets: np.ndarray, pad_id: int) -> np.ndarray:
    """Bool mask [B, T]: True before first pad; False at pad and after."""
    x = np.asarray(targets)
    if x.ndim != 2:
        raise ValueError(f"targets must be 2D [B, T], got shape {x.shape}")
    is_pad = x == int(pad_id)
    seen_pad = np.cumsum(is_pad, axis=1) > 0
    return ~seen_pad


def murcko_scaffold_smiles(mol) -> str | None:
    """Bemis–Murcko scaffold SMILES, or None if unavailable."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


class ScaffoldDiversityFilter:
    """REINVENT-style Murcko bucket memory + exact-SMILES penalty."""

    def __init__(
        self,
        *,
        bucket_size: int = 25,
        penalty: float = 0.0,
        min_score: float = float("-inf"),
    ):
        self.bucket_size = max(int(bucket_size), 1)
        self.penalty = float(penalty)
        self.min_score = float(min_score)
        self._counts: dict[str, int] = {}
        self._smiles: set[str] = set()

    def factor(self, mol, canonical_smiles: str) -> float:
        if canonical_smiles in self._smiles:
            return self.penalty
        scaf = murcko_scaffold_smiles(mol)
        if scaf is not None and self._counts.get(scaf, 0) >= self.bucket_size:
            return self.penalty
        return 1.0

    def remember(self, mol, canonical_smiles: str, score: float) -> None:
        if not math.isfinite(score) or score < self.min_score:
            return
        self._smiles.add(canonical_smiles)
        scaf = murcko_scaffold_smiles(mol)
        if scaf is not None:
            self._counts[scaf] = self._counts.get(scaf, 0) + 1

    def n_scaffolds(self) -> int:
        return len(self._counts)


class EliteReplayBuffer:
    """Keep top-K molecules by quality (pre-DF) for inception-style replay."""

    def __init__(self, capacity: int = 128):
        self.capacity = max(int(capacity), 1)
        self._items: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._items)

    def offer(self, smiles: str, quality: float) -> None:
        if not smiles or not math.isfinite(quality):
            return
        prev = self._items.get(smiles)
        if prev is None or float(quality) > prev:
            self._items[smiles] = float(quality)
        if len(self._items) > self.capacity:
            ranked = sorted(self._items.items(), key=lambda kv: -kv[1])
            self._items = dict(ranked[: self.capacity])

    def sample(
        self,
        k: int,
        rng: np.random.Generator | None = None,
        *,
        weighted: bool = False,
    ) -> list[tuple[str, float]]:
        if k <= 0 or not self._items:
            return []
        rng = rng or np.random.default_rng()
        items = list(self._items.items())
        k = min(int(k), len(items))
        if not weighted or k == len(items):
            idx = rng.choice(len(items), size=k, replace=False)
        else:
            qs = np.asarray([float(q) for _, q in items], dtype=np.float64)
            w = qs - float(np.min(qs)) + 1e-6
            w = w / w.sum()
            idx = rng.choice(len(items), size=k, replace=False, p=w)
        return [items[int(i)] for i in idx]

    def to_list(self) -> list[tuple[str, float]]:
        return sorted(self._items.items(), key=lambda kv: -kv[1])

    def load(self, items: list[tuple[str, float]]) -> None:
        self._items = {}
        for smi, q in items:
            self.offer(str(smi), float(q))


def inception_replay_k(
    base_batch: int,
    boost_batch: int,
    *,
    boost_active: bool,
    n_on_policy: int,
    max_frac: float,
) -> int:
    """
    How many elites to mix into the learn batch.

    Caps so elite_count / (n_on_policy + elite_count) <= max_frac.
    """
    desired = int(boost_batch if boost_active else base_batch)
    if desired <= 0 or int(n_on_policy) <= 0:
        return 0
    frac = min(max(float(max_frac), 0.0), 0.99)
    k_cap = int(int(n_on_policy) * frac / max(1.0 - frac, 1e-9))
    return max(0, min(desired, k_cap))
