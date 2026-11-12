"""Fast SMILES-to-CO2 interaction-energy surrogate.

The model uses inexpensive RDKit 2D descriptors plus a Morgan fingerprint.
Training artifacts are joblib payloads so the same feature contract is used
by offline evaluation and the RL reward scorer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors

DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumRings",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "NumHeteroatoms",
    "FormalCharge",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "BalabanJ",
)

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
FEATURE_SET_LEGACY = "descriptor_morgan_v1"
FEATURE_SET_BROAD = "broad_rdkit_v1"
FEATURE_SET_3D = "broad_rdkit_3d_v1"
FEATURE_SET_CHEMBERTA = "chemberta_v1"
THREE_D_CONFORMERS = 5
THREE_D_DESCRIPTOR_NAMES = (
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "PMI1",
    "PMI2",
    "PMI3",
    "RadiusOfGyration",
    "SpherocityIndex",
    "PBF",
)
_MORGAN_GENERATOR = AllChem.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)
_MORGAN3_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
_RDKIT_GENERATOR = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
_ATOM_PAIR_GENERATOR = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
_TORSION_GENERATOR = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
_BROAD_DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors.descList)


def _descriptor_values(mol: Chem.Mol) -> list[float]:
    values = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        Descriptors.TPSA(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.HeavyAtomCount(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        Chem.GetFormalCharge(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Descriptors.BalabanJ(mol),
    ]
    return [float(v) if np.isfinite(v) else 0.0 for v in values]


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _broad_descriptor_values(mol: Chem.Mol) -> list[float]:
    values = []
    for _, descriptor in Descriptors.descList:
        try:
            value = float(descriptor(mol))
        except Exception:
            value = 0.0
        values.append(value if np.isfinite(value) else 0.0)
    return values


def _fp_array(fp) -> np.ndarray:
    bits = np.zeros(fp.GetNumBits(), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, bits)
    return bits


def _three_d_values(mol: Chem.Mol) -> list[float]:
    """Aggregate deterministic ETKDG conformer descriptors."""
    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 17
    params.pruneRmsThresh = 0.5
    try:
        conformers = list(
            AllChem.EmbedMultipleConfs(
                mol_h, numConfs=THREE_D_CONFORMERS, params=params
            )
        )
    except Exception:
        conformers = []
    if not conformers:
        return [0.0] * (len(THREE_D_DESCRIPTOR_NAMES) * 4)
    calculators = (
        rdMolDescriptors.CalcAsphericity,
        rdMolDescriptors.CalcEccentricity,
        rdMolDescriptors.CalcInertialShapeFactor,
        rdMolDescriptors.CalcNPR1,
        rdMolDescriptors.CalcNPR2,
        rdMolDescriptors.CalcPMI1,
        rdMolDescriptors.CalcPMI2,
        rdMolDescriptors.CalcPMI3,
        rdMolDescriptors.CalcRadiusOfGyration,
        rdMolDescriptors.CalcSpherocityIndex,
        rdMolDescriptors.CalcPBF,
    )
    values: list[float] = []
    for calculator in calculators:
        per_conf = []
        for conf_id in conformers:
            try:
                value = float(calculator(mol_h, confId=int(conf_id)))
            except Exception:
                value = 0.0
            if np.isfinite(value):
                per_conf.append(value)
        if not per_conf:
            per_conf = [0.0]
        values.extend(
            (
                float(np.mean(per_conf)),
                float(np.std(per_conf)),
                float(np.min(per_conf)),
                float(np.max(per_conf)),
            )
        )
    return values


def featurize_mol(mol: Chem.Mol, feature_set: str = FEATURE_SET_LEGACY) -> np.ndarray:
    """Return a deterministic descriptor/fingerprint feature row."""
    desc = np.asarray(_descriptor_values(mol), dtype=np.float32)
    if feature_set == FEATURE_SET_LEGACY:
        return np.concatenate((desc, _fp_array(_MORGAN_GENERATOR.GetFingerprint(mol))))
    if feature_set not in (FEATURE_SET_BROAD, FEATURE_SET_3D):
        raise ValueError(f"unknown feature set: {feature_set}")
    broad = [_broad_descriptor_values(mol)]
    if feature_set == FEATURE_SET_3D:
        broad.append(_three_d_values(mol))
    broad.extend(
        [
            _fp_array(_MORGAN_GENERATOR.GetFingerprint(mol)),
            _fp_array(_MORGAN3_GENERATOR.GetFingerprint(mol)),
            _fp_array(MACCSkeys.GenMACCSKeys(mol)),
            _fp_array(_RDKIT_GENERATOR.GetFingerprint(mol)),
            _fp_array(_ATOM_PAIR_GENERATOR.GetFingerprint(mol)),
            _fp_array(_TORSION_GENERATOR.GetFingerprint(mol)),
        ]
    )
    return np.concatenate([np.asarray(broad[0], dtype=np.float32), *broad[1:]])


def featurize_smiles(
    smiles: str, feature_set: str = FEATURE_SET_LEGACY
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    return None if mol is None else featurize_mol(mol, feature_set)


def feature_names(feature_set: str = FEATURE_SET_LEGACY) -> list[str]:
    if feature_set == FEATURE_SET_CHEMBERTA:
        return []
    if feature_set in (FEATURE_SET_BROAD, FEATURE_SET_3D):
        names = list(_BROAD_DESCRIPTOR_NAMES)
        if feature_set == FEATURE_SET_3D:
            names.extend(
                f"3d_{name}_{stat}"
                for name in THREE_D_DESCRIPTOR_NAMES
                for stat in ("mean", "std", "min", "max")
            )
        for prefix, size in (
            ("morgan2", 2048),
            ("morgan3", 2048),
            ("maccs", 167),
            ("rdkit", 2048),
            ("atompair", 2048),
            ("torsion", 2048),
        ):
            names.extend(f"{prefix}_{i:04d}" for i in range(size))
        return names
    if feature_set != FEATURE_SET_LEGACY:
        raise ValueError(f"unknown feature set: {feature_set}")
    return list(DESCRIPTOR_NAMES) + [
        f"morgan_{i:04d}" for i in range(MORGAN_BITS)
    ]


class IESurrogate:
    """Loaded XGBoost IE surrogate with the persisted feature contract."""

    def __init__(self, payload: dict):
        self.model = payload["model"]
        self.models = list(payload.get("models", [self.model]))
        self.feature_names = list(payload["feature_names"])
        self.feature_set = str(payload.get("feature_set", FEATURE_SET_LEGACY))
        self.model_name = str(payload.get("model_name", "DeepChem/ChemBERTa-77M-MLM"))
        names = payload.get("all_feature_names")
        self.all_feature_names = list(
            names if names is not None else feature_names(self.feature_set)
        )
        self.feature_indices = np.asarray(
            payload.get("feature_indices", np.arange(len(self.all_feature_names))),
            dtype=np.int64,
        )
        self.descriptor_names = list(
            payload.get(
                "descriptor_names",
                DESCRIPTOR_NAMES if self.feature_set == FEATURE_SET_LEGACY else _BROAD_DESCRIPTOR_NAMES,
            )
        )
        self.descriptor_mean = np.asarray(payload["descriptor_mean"], dtype=np.float32)
        self.descriptor_scale = np.asarray(payload["descriptor_scale"], dtype=np.float32)
        self.target_mean = float(payload["target_mean"])
        self.target_scale = float(payload["target_scale"])
        self.uncertainty_scale = float(payload.get("uncertainty_scale", 1.0))
        self.uncertainty_floor = float(payload.get("uncertainty_floor", 0.0))
        self.model_path: Path | None = None
        self._chemberta_encoder = None

    @classmethod
    def load(cls, path: str | Path) -> "IESurrogate":
        model = cls(joblib.load(path))
        model.model_path = Path(path)
        return model

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32).copy()
        n_desc = len(self.descriptor_names)
        if n_desc:
            X[:, :n_desc] = (X[:, :n_desc] - self.descriptor_mean) / self.descriptor_scale
        return X[:, self.feature_indices]

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        pred_z, _ = self.predict_features_with_uncertainty(X)
        return pred_z

    def predict_features_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        transformed = self._transform(X)
        predictions = np.asarray(
            [model.predict(transformed) for model in self.models], dtype=np.float64
        )
        predictions = predictions * self.target_scale + self.target_mean
        spread = np.maximum(
            predictions.std(axis=0) * self.uncertainty_scale,
            self.uncertainty_floor,
        )
        return predictions.mean(axis=0), spread

    def predict(self, smiles: Iterable[str]) -> np.ndarray:
        predictions, _ = self.predict_with_uncertainty(smiles)
        return predictions

    def predict_with_uncertainty(
        self, smiles: Iterable[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        smiles = list(smiles)
        if self.feature_set == FEATURE_SET_CHEMBERTA:
            if self._chemberta_encoder is None:
                from transformers import AutoModel, AutoTokenizer
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                model_name = self.model_name
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                encoder = AutoModel.from_pretrained(model_name).eval().to(device)
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
                self._chemberta_encoder = (tokenizer, encoder, device)
            tokenizer, encoder, device = self._chemberta_encoder
            import torch

            embeddings = np.zeros(
                (len(smiles), int(encoder.config.hidden_size)), dtype=np.float32
            )
            for start in range(0, len(smiles), 64):
                chunk = smiles[start : start + 64]
                tokens = tokenizer(
                    chunk, padding=True, truncation=True, max_length=256,
                    return_tensors="pt"
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                with torch.no_grad():
                    hidden = encoder(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                    pooled = torch.nn.functional.normalize(pooled, dim=-1)
                embeddings[start : start + len(chunk)] = pooled.cpu().numpy()
            return self.predict_features_with_uncertainty(embeddings)
        rows = [featurize_smiles(s, self.feature_set) for s in smiles]
        if any(row is None for row in rows):
            raise ValueError("all SMILES must be valid before prediction")
        return self.predict_features_with_uncertainty(np.stack(rows))
