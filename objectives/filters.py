from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from transpharmer.SMILEValidifier import atom_check


@dataclass(frozen=True)
class FilterConfig:
    """Hard filters ported from `Final_Molecule_Gen/rl.py`."""

    mw_max: float = 500.0
    logp_min: float = 1.0
    logp_max: float = 5.0
    rot_bonds_max: int = 15
    hba_max: int = 10
    hbd_max: int = 5
    sa_max: float = 3.0


def mol_from_smiles(smi: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smi)


def passes_filters(
    smi: str,
    *,
    mol: Chem.Mol | None = None,
    sa_score: float | None = None,
    cfg: FilterConfig = FilterConfig(),
) -> bool:
    """
    Return True iff the molecule passes the same gate used before scoring BA in the old RL code.

    Old gate (from `Final_Molecule_Gen/rl.py`):
      - atom_check
      - MW <= 500
      - 1 <= logP <= 5
      - rot_bonds <= 15
      - HBA <= 10
      - HBD <= 5
      - SA <= 3
    """
    if not atom_check(smi):
        return False

    m = mol if mol is not None else Chem.MolFromSmiles(smi)
    if m is None:
        return False

    mw = float(Descriptors.MolWt(m))
    logp = float(Descriptors.MolLogP(m))
    rot_bonds = int(Lipinski.NumRotatableBonds(m))
    hba = int(Lipinski.NumHAcceptors(m))
    hbd = int(Lipinski.NumHDonors(m))

    if mw > cfg.mw_max:
        return False
    if not (cfg.logp_min <= logp <= cfg.logp_max):
        return False
    if rot_bonds > cfg.rot_bonds_max:
        return False
    if hba > cfg.hba_max:
        return False
    if hbd > cfg.hbd_max:
        return False

    if sa_score is not None and sa_score > cfg.sa_max:
        return False

    return True

