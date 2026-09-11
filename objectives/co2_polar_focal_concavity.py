from __future__ import annotations

"""
CO2 objective component: Polar Focal Concavity (PFC).

Ported from `CO2_Binding_Gen/PolarFocalConcavity/PolarFocalConcavity.py` and placed in
`Molecule_Generation_Workflow/objectives/` so all migrated objectives live there.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.rdchem import Atom, Conformer, Mol
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Soft anchors for mapping raw focal score to (0, 1) via sigmoid (not hard clip).
# Tuned for hardness–dipole S_i on built-in __main__ SMILES (raw roughly 0.05–8.5).
_FOCAL_RAW_LO_HI_DEFAULT: tuple[float, float] = (0.0, 9.0)

# Process-level cache: (canonical_smiles, n_conformers, voxel_step) -> score dict
_PFC_CACHE: dict[tuple[str, int, float], dict[str, float]] = {}


def clear_pfc_cache() -> None:
    _PFC_CACHE.clear()


def pfc_cache_size() -> int:
    return len(_PFC_CACHE)


def _smiles_seed(canonical_smiles: str, fallback: int) -> int:
    """Stable positive 31-bit seed from canonical SMILES."""
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF or int(fallback)

# vdW radii (Å) — same scale as original PFC hull-voxel carve-out
_VDW_RADII: dict[int, float] = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}
_DEFAULT_VDW = 1.70


POLARIZABILITY_ATOM_DICT = {
    "N": 1.096532,
    "O": 0.785354,
    "S": 2.874692,
    "P": 3.7045,
    "F": 0.5541932,
    "Cl": 2.163428,
    "Br": 3.11178,
    "I": 4.875122,
    "C": 1.674434,
    "H": 0.66784726,
}

ELECTRONEGATIVITY_ATOM_DICT = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Br": 2.96,
    "I": 2.66,
    "Si": 1.90,
    "Se": 2.55,
}


@dataclass
class ConcavityVizData:
    positions: np.ndarray  # (N, 3)
    symbols: list[str]
    electronegativity: np.ndarray  # (N,) Pauling, nan if unknown element
    hull: ConvexHull | None
    filled_centers: np.ndarray  # (M, 3) voxel centres inside hull, inside vdW
    hollow_centers: np.ndarray  # (H, 3) voxel centres inside hull, void
    step: float


class PolarFocalConcavity:
    def __init__(
        self,
        w_focal: float = 0.40,
        w_concavity: float = 0.40,
        w_entropy: float = 0.20,  # kept for API compatibility (not used in composite here)
        cluster_target_radius: float = 4.0,
        conformer_seed: int = 42,
        n_conformers: int = 2,
        *,
        concavity_voxel_step: float = 0.5,
        concavity_sigmoid_k: float = 8.0,
        focal_raw_lo: float | None = None,
        focal_raw_hi: float | None = None,
        focal_sigmoid_k: float = 8.0,
        # RDKit EmbedMultipleConfs: 0 = all cores; set 1 in process-pool workers.
        embed_num_threads: int = 0,
    ):
        self.w_focal = w_focal
        self.w_concavity = w_concavity
        self.w_entropy = w_entropy
        self.cluster_target_radius = cluster_target_radius
        self.conformer_seed = conformer_seed
        self.n_conformers = n_conformers
        self.concavity_voxel_step = concavity_voxel_step
        self.concavity_sigmoid_k = concavity_sigmoid_k
        lo, hi = _FOCAL_RAW_LO_HI_DEFAULT
        self.focal_raw_lo = focal_raw_lo if focal_raw_lo is not None else lo
        self.focal_raw_hi = focal_raw_hi if focal_raw_hi is not None else hi
        self.focal_sigmoid_k = focal_sigmoid_k
        self.embed_num_threads = int(embed_num_threads)

    @staticmethod
    def _focal_raw_to_unit(raw: float, lo: float, hi: float, k: float) -> float:
        span = max(hi - lo, 1e-9)
        t = (raw - lo) / span
        return float(1.0 / (1.0 + math.exp(-k * (t - 0.5))))

    def _get_polar_atoms(self, mol: Mol) -> list[Atom]:
        return [a for a in mol.GetAtoms() if a.GetSymbol() in POLARIZABILITY_ATOM_DICT]

    @staticmethod
    def _ensure_gasteiger_charges(mol_h: Mol) -> None:
        rdPartialCharges.ComputeGasteigerCharges(mol_h)

    @staticmethod
    def _gasteiger_charges_array(mol_h: Mol) -> np.ndarray:
        """Per-atom Gasteiger charges (_GasteigerCharge); non-finite values become 0."""
        PolarFocalConcavity._ensure_gasteiger_charges(mol_h)
        n = mol_h.GetNumAtoms()
        out = np.zeros(n, dtype=float)
        for atom in mol_h.GetAtoms():
            q = float(atom.GetDoubleProp("_GasteigerCharge"))
            if not math.isfinite(q):
                q = 0.0
            out[atom.GetIdx()] = q
        return out

    def _atom_focal_score_raw(
        self,
        conf: Conformer,
        atom_i: Atom,
        charges: np.ndarray,
    ) -> float:
        """S_i = (chi^2/alpha)*|q_i^-| + sum_bonded |q_i - q_j|/(d_ij^2 + eps)."""
        eps = 1e-6
        eps_alpha = 1e-12
        i = atom_i.GetIdx()
        sym = atom_i.GetSymbol()
        chi = ELECTRONEGATIVITY_ATOM_DICT.get(sym)
        alpha = POLARIZABILITY_ATOM_DICT.get(sym)
        q_i = float(charges[i])
        q_i_neg_mag = max(-q_i, 0.0)
        if chi is None or alpha is None:
            hard_density = 0.0
        else:
            chi_f = float(chi)
            alpha_f = float(alpha)
            if not math.isfinite(chi_f) or not math.isfinite(alpha_f):
                hard_density = 0.0
            else:
                hard_density = (chi_f * chi_f) / max(alpha_f, eps_alpha) * q_i_neg_mag

        score = hard_density
        pos_i = conf.GetAtomPosition(i)
        for neighbor in atom_i.GetNeighbors():
            j = neighbor.GetIdx()
            dist = float(pos_i.Distance(conf.GetAtomPosition(j)))
            den = dist * dist + eps
            q_j = float(charges[j])
            score += abs(q_i - q_j) / den
        return float(score)

    def compute_focal_score(
        self,
        mol_h: Mol,
        conf: Conformer,
        gasteiger_charges: np.ndarray | None = None,
    ) -> float:
        """Best polar-center S_i for a single conformer (hardness–charge + bonded dipole tension)."""
        charges = (
            gasteiger_charges
            if gasteiger_charges is not None
            else self._gasteiger_charges_array(mol_h)
        )
        polar_atoms = self._get_polar_atoms(mol_h)
        best_score = 0.0
        for atom in polar_atoms:
            best_score = max(
                best_score,
                self._atom_focal_score_raw(conf, atom, charges),
            )
        return float(best_score)

    def per_atom_focal_scores_raw(
        self,
        mol_h: Mol,
        conf: Conformer,
        gasteiger_charges: np.ndarray | None = None,
    ) -> np.ndarray:
        """Length-N array: S_i for polar focal centers, else 0 (for viz)."""
        charges = (
            gasteiger_charges
            if gasteiger_charges is not None
            else self._gasteiger_charges_array(mol_h)
        )
        n = mol_h.GetNumAtoms()
        out = np.zeros(n, dtype=float)
        for atom in mol_h.GetAtoms():
            if atom.GetSymbol() not in POLARIZABILITY_ATOM_DICT:
                continue
            idx = atom.GetIdx()
            out[idx] = self._atom_focal_score_raw(conf, atom, charges)
        return out

    @staticmethod
    def _conformer_positions(mol_h: Mol, conf: Conformer) -> np.ndarray:
        try:
            return np.asarray(conf.GetPositions(), dtype=float)
        except AttributeError:
            n = mol_h.GetNumAtoms()
            coords = np.zeros((n, 3), dtype=float)
            for i in range(n):
                p = conf.GetAtomPosition(i)
                coords[i] = (p.x, p.y, p.z)
            return coords

    # Chunk size for vectorized voxel tests (bounds peak RAM).
    _VOXEL_CHUNK = 65536

    @staticmethod
    def _voxel_grid_classification(
        positions: np.ndarray,
        atoms: list[Atom],
        step: float,
        *,
        collect_centers: bool = False,
    ) -> tuple[float, ConvexHull | None, np.ndarray, np.ndarray]:
        empty = np.zeros((0, 3), dtype=float)
        n = positions.shape[0]
        if n < 4:
            return 0.0, None, empty, empty
        try:
            hull = ConvexHull(positions)
        except Exception:
            return 0.0, None, empty, empty

        eqs = hull.equations
        radii = np.array(
            [_VDW_RADII.get(a.GetAtomicNum(), _DEFAULT_VDW) for a in atoms], dtype=float
        )
        margin = step
        lo = positions.min(axis=0) - margin
        hi = positions.max(axis=0) + margin

        # Same open-ended grid as the old while x < x1; x += step loop.
        xs = np.arange(lo[0], hi[0], step, dtype=float)
        ys = np.arange(lo[1], hi[1], step, dtype=float)
        zs = np.arange(lo[2], hi[2], step, dtype=float)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            return 0.0, hull, empty, empty

        nx, ny, nz = xs.size, ys.size, zs.size
        n_total = nx * ny * nz
        hull_voxels = 0
        hollow_voxels = 0
        filled_chunks: list[np.ndarray] = []
        hollow_chunks: list[np.ndarray] = []

        normals = eqs[:, :3]
        offsets = eqs[:, 3]
        chunk = PolarFocalConcavity._VOXEL_CHUNK
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            idx = np.arange(start, end)
            # Flattened order matches nested x-outer, then y, then z (old triple loop).
            ix = idx // (ny * nz)
            rem = idx % (ny * nz)
            iy = rem // nz
            iz = rem % nz
            pts = np.column_stack((xs[ix], ys[iy], zs[iz]))

            # Outside hull if any halfspace constraint is violated.
            outside = np.any(normals @ pts.T + offsets[:, None] > 1e-10, axis=0)
            inside = ~outside
            if not np.any(inside):
                continue
            pts_in = pts[inside]
            hull_voxels += int(pts_in.shape[0])

            filled_mask = (cdist(pts_in, positions) <= radii[None, :]).any(axis=1)
            hollow_mask = ~filled_mask
            hollow_voxels += int(hollow_mask.sum())
            if collect_centers:
                if np.any(filled_mask):
                    filled_chunks.append(pts_in[filled_mask])
                if np.any(hollow_mask):
                    hollow_chunks.append(pts_in[hollow_mask])

        ratio = 0.0 if hull_voxels == 0 else hollow_voxels / hull_voxels
        if collect_centers:
            filled = np.vstack(filled_chunks) if filled_chunks else empty
            hollow = np.vstack(hollow_chunks) if hollow_chunks else empty
        else:
            filled, hollow = empty, empty
        return float(ratio), hull, filled, hollow

    def calculate_concavity_score(self, mol_h: Mol, conf: Conformer) -> float:
        coords = self._conformer_positions(mol_h, conf)
        atoms = list(mol_h.GetAtoms())
        if coords.shape[0] != len(atoms):
            return 0.0

        ratio, _, _, _ = self._voxel_grid_classification(
            coords, atoms, self.concavity_voxel_step, collect_centers=False
        )
        k = self.concavity_sigmoid_k
        norm_concavity = float(1.0 - math.exp(-k * ratio))
        return norm_concavity

    def get_concavity_visualization_data(self, mol_h: Mol, conf: Conformer) -> ConcavityVizData:
        coords = self._conformer_positions(mol_h, conf)
        atoms = list(mol_h.GetAtoms())
        if coords.shape[0] != len(atoms):
            raise ValueError("Atom count does not match conformer coordinates.")

        step = self.concavity_voxel_step
        _, hull, filled, hollow = self._voxel_grid_classification(
            coords, atoms, step, collect_centers=True
        )

        symbols = [a.GetSymbol() for a in atoms]
        en = np.array(
            [float(ELECTRONEGATIVITY_ATOM_DICT.get(sym, float("nan"))) for sym in symbols],
            dtype=float,
        )
        return ConcavityVizData(
            positions=coords.copy(),
            symbols=symbols,
            electronegativity=en,
            hull=hull,
            filled_centers=filled,
            hollow_centers=hollow,
            step=step,
        )

    def compute(self, mol_input: str | Mol) -> dict[str, float]:
        can: str | None = None
        if isinstance(mol_input, str):
            mol = Chem.MolFromSmiles(mol_input)
            if mol is None:
                raise ValueError(f"Could not parse SMILES: {mol_input!r}")
            can = Chem.MolToSmiles(mol)
        else:
            mol = mol_input
            try:
                can = Chem.MolToSmiles(mol)
            except Exception:
                can = None

        cache_key: tuple[str, int, float] | None = None
        if can is not None:
            cache_key = (can, int(self.n_conformers), float(self.concavity_voxel_step))
            hit = _PFC_CACHE.get(cache_key)
            if hit is not None and "min_nn_distance" in hit:
                return dict(hit)

        seed = (
            _smiles_seed(can, self.conformer_seed)
            if can is not None
            else int(self.conformer_seed)
        )

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = int(self.embed_num_threads)
        conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=self.n_conformers, params=params)
        if not conf_ids:
            out = {
                "focal_score": 0.0,
                "focal_score_raw": 0.0,
                "concavity_score": 0.0,
                "composite": 0.0,
                "min_nn_distance": float("nan"),
            }
            if cache_key is not None:
                _PFC_CACHE[cache_key] = dict(out)
            return out

        from objectives.co2_rl_helpers import min_nonbonded_nn_distance_over_confs

        gasteiger_charges = self._gasteiger_charges_array(mol_h)
        focal_raws: list[float] = []
        concavities: list[float] = []
        for cid in conf_ids:
            conf = mol_h.GetConformer(cid)
            focal_raws.append(self.compute_focal_score(mol_h, conf, gasteiger_charges))
            concavities.append(self.calculate_concavity_score(mol_h, conf))

        mean_focal_raw = float(np.mean(focal_raws))
        mean_concavity = float(np.mean(concavities))
        focal_u = self._focal_raw_to_unit(
            mean_focal_raw, self.focal_raw_lo, self.focal_raw_hi, self.focal_sigmoid_k
        )
        composite = self.w_focal * focal_u + self.w_concavity * mean_concavity
        out = {
            "focal_score": float(focal_u),
            "focal_score_raw": float(mean_focal_raw),
            "concavity_score": float(mean_concavity),
            "composite": float(composite),
            # Reuse the same ETKDG confs for CO2 min N···N soft penalty.
            "min_nn_distance": float(min_nonbonded_nn_distance_over_confs(mol_h)),
        }
        if cache_key is not None:
            _PFC_CACHE[cache_key] = dict(out)
        return out

