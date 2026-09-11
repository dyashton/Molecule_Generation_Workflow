"""
Standalone SCScorer model (numpy + RDKit).

Ported from `Final_Molecule_Gen/SAScoringFunction/standalone_model_numpy.py` and kept
inside `objectives/` so all migrated objectives live in the objectives folder.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

score_scale = 5.0
FP_len = 1024
FP_rad = 2


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class SCScorer:
    def __init__(self, score_scale: float = score_scale):
        self.vars = []
        self.score_scale = score_scale
        self._restored = False

    def restore(
        self,
        weight_path: str | Path,
        *,
        FP_rad: int = FP_rad,
        FP_len: int = FP_len,
    ):
        self.FP_len = FP_len
        self.FP_rad = FP_rad
        self._load_vars(str(weight_path))

        weight_path_s = str(weight_path)
        if "uint8" in weight_path_s or "counts" in weight_path_s:

            def mol_to_fp(self, mol):
                if mol is None:
                    return np.array((self.FP_len,), dtype=np.uint8)
                fp = AllChem.GetMorganFingerprint(mol, self.FP_rad, useChirality=True)
                fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
                for k, v in fp.GetNonzeroElements().items():
                    fp_folded[k % self.FP_len] += v
                return np.array(fp_folded)

        else:

            def mol_to_fp(self, mol):
                if mol is None:
                    return np.zeros((self.FP_len,), dtype=np.float32)
                return np.array(
                    AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.FP_rad, nBits=self.FP_len, useChirality=True
                    ),
                    dtype=bool,
                )

        self.mol_to_fp = mol_to_fp
        self._restored = True
        return self

    def smi_to_fp(self, smi: str):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def apply(self, x):
        if not self._restored:
            raise ValueError("Must restore model weights!")
        for i in range(0, len(self.vars), 2):
            last_layer = i == len(self.vars) - 2
            W = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU
        x = 1 + (score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi: str = "", v: bool = False):
        if not smi:
            return ("", 0.0)
        fp = np.array((self.smi_to_fp(smi)), dtype=np.float32)
        if float(np.sum(fp)) == 0.0:
            if v:
                print("Could not get fingerprint?")
            cur_score = 0.0
        else:
            cur_score = self.apply(fp)
            if v:
                print(f"Score: {cur_score}")
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ""
        return (smi, cur_score)

    def _load_vars(self, weight_path: str):
        if weight_path.endswith("json.gz"):
            with gzip.GzipFile(weight_path, "r") as fin:
                json_bytes = fin.read()
                json_str = json_bytes.decode("utf-8")
                self.vars = json.loads(json_str)
                self.vars = [np.array(x) for x in self.vars]
        else:
            raise ValueError(f"Unsupported weight format: {weight_path}")

