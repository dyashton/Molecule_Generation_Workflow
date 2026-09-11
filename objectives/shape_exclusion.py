from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree


def _here() -> Path:
    return Path(__file__).resolve().parent


class ShapeExclusionScreener:
    """
    Port of `Final_Molecule_Gen/transpharmer/shapeExclusion.py`.

    Uses a receptor structure (`8ETR.pdb`) and checks if any sampled pose of a generated
    ligand avoids clashes with receptor atoms (simple collision spheres).
    """

    def __init__(self, pdb_file: str | Path | None = None):
        pdb = Path(pdb_file) if pdb_file is not None else (_here() / "assets" / "8ETR.pdb")
        self.pdb_file = pdb
        self.structure = self.load_pdb(self.pdb_file)
        self.ligand_coords, self.receptor_coords = self.get_coords(
            self.structure, ligand_resname="WTN"
        )
        self.ligand_center = np.mean(self.ligand_coords, axis=0)
        self.tree = cKDTree(self.receptor_coords)

    def smiles_to_3d(self, smiles: str, num_confs: int = 20):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        mol = Chem.AddHs(mol)

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
        if not conf_ids:
            return [], mol, conf_ids

        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

        all_coords = []
        for conf_id in conf_ids:
            conf = mol.GetConformer(conf_id)
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            all_coords.append(np.array(coords))
        return all_coords, mol, conf_ids

    def load_pdb(self, file_path: str | Path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("PDB_structure", str(file_path))
        return structure

    def get_coords(self, structure, ligand_resname: str = "WTN"):
        ligand_coords = []
        receptor_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if residue.get_resname().strip() == ligand_resname:
                            ligand_coords.append(atom.get_coord())
                        else:
                            receptor_coords.append(atom.get_coord())
        return np.array(ligand_coords), np.array(receptor_coords)

    @staticmethod
    def position_lig_molecule(mol_coords, true_ligand_center):
        mol_center = mol_coords.mean(axis=0)
        translation = true_ligand_center - mol_center
        return mol_coords + translation

    @staticmethod
    def fits_collision_spheres(lig_coords, tree, lig_radii=None, rec_radii: float = 0.5):
        if lig_radii is None:
            lig_radii = np.full(lig_coords.shape[0], 0.5)

        clash_count = 0
        for lig_atom, lig_radius in zip(lig_coords, lig_radii):
            indices = tree.query_ball_point(lig_atom, r=float(lig_radius + rec_radii))
            if indices:
                clash_count += len(indices)
        fits = clash_count == 0
        return fits

    @staticmethod
    def rotation_matrix_from_axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c
        R = np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ]
        )
        return R

    def generate_poses(self, all_coords, max_translation: float = 0.5, num_poses: int = 200):
        new_coords = []
        for coords in all_coords:
            for _ in range(num_poses):
                translation = np.random.uniform(-max_translation, max_translation, size=3)
                angle = np.random.uniform(0, 2 * np.pi)
                axis = np.random.normal(size=3)
                axis /= np.linalg.norm(axis)
                rotation_matrix = self.rotation_matrix_from_axis_angle(axis, angle)
                centered = coords - coords.mean(axis=0)
                pose = centered @ rotation_matrix.T + self.ligand_center + translation
                new_coords.append(pose)
        return new_coords

    def single_smiles_shape_exclusion_check(self, smiles: str) -> bool:
        coords_aligned = []
        conformations_coords_list, _mol, _conf_ids = self.smiles_to_3d(smiles)
        for conformation in conformations_coords_list:
            coords_aligned.append(self.position_lig_molecule(conformation, self.ligand_center))

        poses = self.generate_poses(coords_aligned)
        for pose in poses:
            if self.fits_collision_spheres(pose, self.tree):
                return True
        return False

