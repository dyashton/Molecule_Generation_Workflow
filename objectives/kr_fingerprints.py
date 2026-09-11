from __future__ import annotations

import json
import numbers
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, MolStandardize
from rdkit.Chem.Fingerprints import FingerprintMols


def _here() -> Path:
    return Path(__file__).resolve().parent


def _krfp_json_path() -> Path:
    return _here() / "assets" / "binding_affinity" / "krfp.json"


class KRFingerprints:
    """
    Port of `Final_Molecule_Gen/ScoringFunction/KRFingerprints.py`.

    NOTE: This relies on `krfp.json` living under `objectives/assets/binding_affinity/`.
    """

    with open(_krfp_json_path(), "r") as json_file:
        KRFPDictSmarts = json.load(json_file)

    KRFPKeys = list(KRFPDictSmarts.keys())

    @staticmethod
    def SmartsToMol(KRFPDictSmarts, KRFPKeys):
        return {x: Chem.MolFromSmarts(KRFPDictSmarts[x]) for x in KRFPKeys}

    KRFPDictMol = SmartsToMol(KRFPDictSmarts, KRFPKeys)

    @staticmethod
    def GenerateKRFingerprints(structures, count=False, output_type="list", verbose=True):
        if isinstance(structures, str) or isinstance(structures, Chem.rdchem.Mol):
            structures = [structures]

        krfp_ligands = []

        RDLogger.DisableLog("rdApp.info")
        for i, ligand in enumerate(structures):
            start = time.time()
            if isinstance(ligand, str):
                ligand = Chem.MolFromSmiles(ligand)
            ligand = MolStandardize.rdMolStandardize.Cleanup(ligand)
            ligand = MolStandardize.rdMolStandardize.FragmentParent(ligand)
            uncharger = MolStandardize.rdMolStandardize.Uncharger()
            ligand = uncharger.uncharge(ligand)

            if count is False:
                krfp_ligand = [
                    1 if ligand.HasSubstructMatch(KRFingerprints.KRFPDictMol[fp]) else 0
                    for fp in KRFingerprints.KRFPDictMol
                ]
            else:
                krfp_ligand = [
                    len(ligand.GetSubstructMatches(KRFingerprints.KRFPDictMol[fp]))
                    for fp in KRFingerprints.KRFPDictMol
                ]
            krfp_ligands.append(krfp_ligand)

            end = time.time()
            if verbose:
                estimated_time = (end - start) * (len(structures) - i + 1)
                estimated_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time))
                print(
                    f"{i+1}/{len(structures)} structures, time remaining: {estimated_time_str}"
                )
                percentage = (i + 1) / len(structures)
                for _ in range(int(percentage * 20)):
                    print("#", end="")
                for _ in range(20 - int(percentage * 20)):
                    print(" ", end="")
                print(f" {int(percentage*100)}%")

        if output_type == "list":
            return krfp_ligands
        if output_type == "dictionary":
            return KRFingerprints.ListToDictionary(krfp_ligands)
        if output_type == "dataframe":
            return pd.DataFrame(KRFingerprints.ListToDictionary(krfp_ligands))
        return krfp_ligands

    @staticmethod
    def GenerateKRFingerprintsToDataFrame(
        data, structures_column, count=False, verbose=True
    ):
        krfp_ligands = KRFingerprints.GenerateKRFingerprints(
            data[structures_column], count, "dictionary", verbose=verbose
        )
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    krfp_ligands,
                    index=data.index,
                    columns=list(KRFingerprints.KRFPDictSmarts.keys()),
                ),
            ],
            axis=1,
        )
        return data

    @staticmethod
    def ListToDictionary(krfp_list):
        descriptors = {}
        for i, ligand_krfp in enumerate(krfp_list):
            for j, fp in enumerate(KRFingerprints.KRFPDictMol):
                if i == 0:
                    descriptors[fp] = [ligand_krfp[j]]
                else:
                    descriptors[fp][i:] = [ligand_krfp[j]]
        return descriptors

    @staticmethod
    def DrawFingerprint(krfp):
        fp = KRFingerprints.KRFPDictMol[krfp]
        img = Draw.MolToImage(fp, legend=krfp)
        return img

    @staticmethod
    def DrawFingerprints(krfps, molsPerRow=3):
        if isinstance(krfps, str):
            krfps = [krfps]
        structs = [KRFingerprints.KRFPDictMol[x] for x in krfps]
        img = Draw.MolsToGridImage(
            structs, molsPerRow=molsPerRow, legends=krfps, subImgSize=(300, 300)
        )
        return img

    @staticmethod
    def FindKRFP(find):
        find = Chem.MolFromSmiles(find)
        find_fps = FingerprintMols.FingerprintMol(find)
        krfp_fps = {
            x: FingerprintMols.FingerprintMol(KRFingerprints.KRFPDictMol[x])
            for x in KRFingerprints.KRFPKeys
        }
        tanimotos = {
            x: DataStructs.TanimotoSimilarity(find_fps, krfp_fps[x])
            for x in KRFingerprints.KRFPKeys
        }
        tanimotos = dict(sorted(tanimotos.items(), key=lambda item: item[1], reverse=True))
        tanimotos_return = {}
        for i in tanimotos:
            if tanimotos[i] == 1.0:
                tanimotos_return[i] = tanimotos[i]
        if len(tanimotos_return) == 0:
            tanimotos_return[list(tanimotos.items())[0][0]] = list(tanimotos.items())[0][1]
        return tanimotos_return

    @staticmethod
    def HighlightKRFP(structures, krfp, names=None):
        if isinstance(structures, Chem.rdchem.Mol):
            structures = [structures]
        if isinstance(names, str):
            names = [names]

        if names is not None:
            for i, name in enumerate(names):
                if isinstance(name, numbers.Number):
                    names[i] = str(name)

        fp = KRFingerprints.KRFPDictMol[krfp]
        mols_atoms = []
        mols_bonds = []

        for i, mol in enumerate(structures):
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                structures[i] = mol

            atoms = sum(mol.GetSubstructMatches(fp), ())
            mols_atoms.append(atoms)

            bonds = []
            for a1 in atoms:
                for a2 in atoms:
                    s = mol.GetBondBetweenAtoms(a1, a2)
                    if s is not None:
                        bonds.append(s.GetIdx())
            mols_bonds.append(bonds)

        img = Draw.MolsToGridImage(
            structures,
            molsPerRow=3,
            legends=names,
            subImgSize=(300, 300),
            highlightAtomLists=mols_atoms,
            highlightBondLists=mols_bonds,
        )
        return img

