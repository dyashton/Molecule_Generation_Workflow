"""Morgan-FP soft floor vs a cached prior SMILES bank (CO2 anti-OOD tether)."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def prior_fp_floor_factor(max_sim: float, floor: float) -> float:
    """Soft multiply: 1 above floor, linear ``sim/floor`` below. ``floor<=0`` → 1."""
    if floor <= 0.0:
        return 1.0
    if max_sim >= floor:
        return 1.0
    return float(max(0.0, max_sim / floor))


def load_smiles_cache(path: Path) -> list[str]:
    """Load unique parseable SMILES from a one-SMILES-per-line file (optional header)."""
    if not path.is_file():
        return []
    out: list[str] = []
    seen: set[str] = set()
    with path.open() as f:
        for line in f:
            smi = line.strip()
            if not smi or smi.upper() == "SMILES":
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol, canonical=True)
            if can in seen:
                continue
            seen.add(can)
            out.append(can)
    return out


def save_smiles_cache(path: Path, smiles: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for smi in smiles:
            f.write(f"{smi}\n")


class PriorFPTether:
    """Max Morgan Tanimoto to a fixed prior bank → soft reward floor factor."""

    def __init__(
        self,
        *,
        floor: float = 0.15,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        self.floor = float(floor)
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self._bank_fps: list = []
        self.n_bank = 0

    @property
    def enabled(self) -> bool:
        return self.floor > 0.0 and self.n_bank > 0

    def set_bank(self, smiles: list[str]) -> int:
        fps = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits))
        self._bank_fps = fps
        self.n_bank = len(fps)
        return self.n_bank

    def max_similarity(self, smiles: str) -> float:
        if not self._bank_fps:
            return 1.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        sims = DataStructs.BulkTanimotoSimilarity(fp, self._bank_fps)
        return float(max(sims)) if sims else 0.0

    def factor(self, smiles: str) -> float:
        if not self.enabled:
            return 1.0
        return prior_fp_floor_factor(self.max_similarity(smiles), self.floor)


if __name__ == "__main__":
    # ponytail: assert-based self-check; no test framework.
    assert abs(prior_fp_floor_factor(0.30, 0.15) - 1.0) < 1e-12
    assert abs(prior_fp_floor_factor(0.15, 0.15) - 1.0) < 1e-12
    assert abs(prior_fp_floor_factor(0.075, 0.15) - 0.5) < 1e-12
    assert abs(prior_fp_floor_factor(0.0, 0.15) - 0.0) < 1e-12
    assert abs(prior_fp_floor_factor(0.05, 0.0) - 1.0) < 1e-12

    tether = PriorFPTether(floor=0.15)
    n = tether.set_bank(["c1ccccc1", "c1ccccc1C", "CCO"])
    assert n == 3
    assert tether.factor("c1ccccc1") == 1.0  # exact bank hit
    # Far from aromatics bank → soft scale (still >0 for tiny organics).
    f_far = tether.factor("C")
    assert 0.0 <= f_far < 1.0, f_far
    print("prior_fp_tether self-check: ok")
