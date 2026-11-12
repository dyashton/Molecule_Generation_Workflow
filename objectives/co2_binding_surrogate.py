"""Hybrid CO2 binding surrogate reward (IE/BE + clamped penalty → BE_GM)."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from objectives.co2_rl_helpers import mw_penalty_excess

# Prefer vendored HPC surrogate; fall back to sibling CO2_IE_Dataset install.
try:
    from co2ie.surrogate import (  # type: ignore
        FEATURE_SET_3D,
        IESurrogate,
        canonicalize_smiles,
        featurize_smiles,
    )
except ImportError:  # pragma: no cover - path bootstrap for bare clone
    import sys

    _vendor = Path(__file__).resolve().parents[1] / "vendor"
    if str(_vendor) not in sys.path:
        sys.path.insert(0, str(_vendor))
    from co2ie.surrogate import (  # type: ignore
        FEATURE_SET_3D,
        IESurrogate,
        canonicalize_smiles,
        featurize_smiles,
    )

_SCORER: "BindingSurrogateScorer | None" = None
ORGANIC_ELEMENTS = frozenset({"C", "H", "N", "O", "F", "S", "Cl", "Br", "I"})


def inorganic_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0
    return sum(atom.GetSymbol() not in ORGANIC_ELEMENTS for atom in mol.GetAtoms())


def charged_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0
    return sum(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def _default_pack_path() -> Path:
    """Local pack first (HPC), then sibling monorepo path."""
    root = Path(__file__).resolve().parents[1]
    local = root / "artifacts" / "binding_predictor" / "pack.joblib"
    if local.is_file():
        return local
    sibling = root.parent / "CO2_IE_Dataset" / "artifacts" / "binding_predictor" / "pack.joblib"
    return sibling


class BindingSurrogateScorer:
    """Reward = -binding_hat with uncertainty + Morgan/Mahalanobis OOD floor."""

    def __init__(
        self,
        pack_path: str | Path | None = None,
        inorganic_penalty: float | None = None,
        charged_penalty: float | None = None,
        mw_penalty_weight: float | None = None,
        strong_bonus: float | None = None,
        tanimoto_floor: float | None = None,
        max_uncertainty: float | None = None,
    ):
        path = Path(
            pack_path
            or os.environ.get("MGW_BINDING_PACK", _default_pack_path())
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Binding predictor pack missing: {path}. "
                "Run CO2_IE_Dataset/scripts/train_binding_predictor.py first."
            )
        self.pack = joblib.load(path)
        self.pack_path = path
        base = path.parent
        self.ie = IESurrogate.load(self.pack.get("ie_path", base / "ie.joblib"))
        self.be = IESurrogate.load(self.pack.get("be_path", base / "be.joblib"))
        self.penalty = IESurrogate.load(
            self.pack.get("penalty_path", base / "penalty.joblib")
        )
        self._penalty_payload = joblib.load(
            self.pack.get("penalty_path", base / "penalty.joblib")
        )
        self.strong = joblib.load(self.pack.get("strong_path", base / "strong.joblib"))
        # HPC contract: pin IE until BE_GM compose clears the published gate.
        # Override with MGW_BINDING_HEAD only when you intentionally change heads.
        default_head = str(self.pack.get("reward_head", "ie"))
        if default_head.startswith("begm") and not os.environ.get("MGW_BINDING_HEAD"):
            default_head = "ie"
        self.reward_head = str(os.environ.get("MGW_BINDING_HEAD", default_head))
        self.inorganic_penalty = float(
            os.environ.get("MGW_INORGANIC_PENALTY", "0.0")
            if inorganic_penalty is None
            else inorganic_penalty
        )
        self.charged_penalty = float(
            os.environ.get("MGW_CHARGED_PENALTY", "0.0")
            if charged_penalty is None
            else charged_penalty
        )
        self.mw_penalty_weight = float(
            os.environ.get("MGW_MW_PENALTY_WEIGHT", "5.0")
            if mw_penalty_weight is None
            else mw_penalty_weight
        )
        self.strong_bonus = float(
            os.environ.get("MGW_BINDING_STRONG_BONUS", "0.0")
            if strong_bonus is None
            else strong_bonus
        )
        ood = self.pack.get("ood", {})
        self.tanimoto_floor = float(
            tanimoto_floor
            if tanimoto_floor is not None
            else ood.get("tanimoto_floor", 0.35)
        )
        raw_u = os.environ.get("MGW_BINDING_MAX_UNCERTAINTY", "").strip()
        if max_uncertainty is not None:
            self.max_uncertainty = float(max_uncertainty)
        elif raw_u:
            self.max_uncertainty = float(raw_u)
        else:
            # Default: 2x OOF RMSE of the active head.
            unc = self.pack.get("uncertainty", {})
            if self.reward_head.startswith("begm"):
                self.max_uncertainty = 2.0 * float(
                    np.hypot(unc.get("be_oof_rmse", 1.0), unc.get("penalty_oof_rmse", 1.0))
                )
            else:
                self.max_uncertainty = 2.0 * float(unc.get("ie_oof_rmse", 1.0))
        self._bank_fps = [
            DataStructs.CreateFromBitString(bits)
            for bits in ood.get("bank_fp_bitstrings", [])
        ]
        self._mahal_idx = np.asarray(
            ood.get("mahal_feature_indices", []), dtype=int
        )
        self._mahal_mean = np.asarray(ood.get("mahal_mean", []), dtype=float)
        self._mahal_cov_inv = np.asarray(ood.get("mahal_cov_inv", []), dtype=float)
        self._desc_mean = np.asarray(ood.get("descriptor_mean", []), dtype=float)
        self._desc_scale = np.asarray(ood.get("descriptor_scale", []), dtype=float)
        self._mahal_cutoff = float(ood.get("mahal_cutoff", np.inf))
        self._cache: dict[str, dict[str, float]] = {}

    def _max_tanimoto(self, smiles: str) -> float:
        if not self._bank_fps:
            return 1.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return float(max(DataStructs.BulkTanimotoSimilarity(fp, self._bank_fps)))

    def _mahalanobis(self, features: np.ndarray) -> float:
        if self._mahal_idx.size == 0 or self._mahal_cov_inv.size == 0:
            return 0.0
        n = len(self._desc_mean)
        z = (features[:n] - self._desc_mean) / self._desc_scale
        zk = z[self._mahal_idx]
        diff = zk - self._mahal_mean
        return float(np.sqrt(max(diff @ self._mahal_cov_inv @ diff, 0.0)))

    def _predict_strong(self, features: np.ndarray) -> float:
        payload = self.strong
        n_desc = len(payload["descriptor_names"])
        x = np.asarray(features, dtype=np.float32).copy()
        x[:n_desc] = (x[:n_desc] - payload["descriptor_mean"]) / payload["descriptor_scale"]
        x = x[np.asarray(payload["feature_indices"], dtype=int)].reshape(1, -1)
        model = payload["model"]
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(x)[0, 1])
        return float(model.predict(x)[0])

    def predict_one(self, smiles: str) -> dict[str, float] | None:
        can = canonicalize_smiles(smiles)
        if can is None:
            return None
        if can in self._cache:
            return self._cache[can]
        features = featurize_smiles(can, FEATURE_SET_3D)
        if features is None or not np.all(np.isfinite(features)):
            return None
        ie_pred, ie_unc = self.ie.predict_with_uncertainty([can])
        be_pred, be_unc = self.be.predict_with_uncertainty([can])
        pen_pred, pen_unc = self.penalty.predict_with_uncertainty([can])
        ie_hat = float(ie_pred[0])
        be_hat = float(be_pred[0])
        pen_hat = float(pen_pred[0])
        # IESurrogate returns z-space→label space but does not invert log1p.
        if self._penalty_payload.get("target_transform") == "log1p":
            pen_hat = float(np.expm1(pen_hat))
        if self._penalty_payload.get("force_nonnegative", True):
            pen_hat = float(max(pen_hat, 0.0))
        begm_hat = be_hat + pen_hat
        strong_p = self._predict_strong(features)
        if self.reward_head.startswith("begm"):
            binding = begm_hat
            uncertainty = float(np.hypot(be_unc[0], pen_unc[0]))
        else:
            binding = ie_hat
            uncertainty = float(ie_unc[0])
        tanimoto = self._max_tanimoto(can)
        mahal = self._mahalanobis(features)
        ood = bool(tanimoto < self.tanimoto_floor or mahal > self._mahal_cutoff)
        result = {
            "ie": ie_hat,
            "be": be_hat,
            "penalty": pen_hat,
            "begm": begm_hat,
            "strong_prob": float(strong_p),
            "binding": binding,
            "uncertainty": uncertainty,
            "tanimoto": tanimoto,
            "mahalanobis": mahal,
            "ood": float(ood),
        }
        self._cache[can] = result
        return result

    def compute_batch(self, smiles_list: list[str]):
        from reward import RewardBatch

        policy: list[float] = []
        raw_binding: list[float] = []
        uncertainties: list[float] = []
        gate_ok: list[bool] = []
        inorganic_counts = [inorganic_atom_count(smi) for smi in smiles_list]
        inorganic_penalties = [
            self.inorganic_penalty * min(count, 1) for count in inorganic_counts
        ]
        charged_counts = [charged_atom_count(smi) for smi in smiles_list]
        charged_penalties = [
            self.charged_penalty * min(count, 1) for count in charged_counts
        ]
        mw_penalties = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi))
            mw_penalties.append(
                self.mw_penalty_weight * mw_penalty_excess(Descriptors.MolWt(mol))
                if mol is not None
                else 0.0
            )
        for i, smi in enumerate(smiles_list):
            pred = self.predict_one(smi)
            ok = (
                pred is not None
                and np.isfinite(pred["binding"])
                and pred["uncertainty"] <= self.max_uncertainty
                and not bool(pred["ood"])
            )
            gate_ok.append(bool(ok))
            raw_binding.append(float(pred["binding"]) if ok and pred else float("nan"))
            uncertainties.append(
                float(pred["uncertainty"]) if pred is not None else float("nan")
            )
            if ok and pred is not None:
                reward = (
                    -pred["binding"]
                    + self.strong_bonus * pred["strong_prob"]
                    - inorganic_penalties[i]
                    - charged_penalties[i]
                    - mw_penalties[i]
                )
                policy.append(float(reward))
            else:
                policy.append(float("nan"))
        return RewardBatch(
            policy=policy,
            raw_total=policy.copy(),
            raw_ba=[float("nan")] * len(smiles_list),
            raw_div=[float("nan")] * len(smiles_list),
            gate_ok=gate_ok,
            raw_physchem=raw_binding,
            quality=policy.copy(),
            mw_scale=[float("nan")] * len(smiles_list),
            surrogate_uncertainty=uncertainties,
            inorganic_atom_count=inorganic_counts,
            inorganic_penalty=inorganic_penalties,
            charged_atom_count=charged_counts,
            charged_penalty=charged_penalties,
            mw_penalty=mw_penalties,
        )


def get_binding_surrogate_scorer() -> BindingSurrogateScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = BindingSurrogateScorer()
    return _SCORER


def configure_binding_surrogate_scorer(**kwargs) -> None:
    global _SCORER
    _SCORER = BindingSurrogateScorer(**kwargs)


def reset_binding_surrogate_scorer() -> None:
    global _SCORER
    _SCORER = None
