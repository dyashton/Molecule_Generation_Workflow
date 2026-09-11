from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem


def _here() -> Path:
    return Path(__file__).resolve().parent


def _assets_dir() -> Path:
    # Kept under objectives/ so everything migrated stays in the objectives folder.
    return _here() / "assets" / "binding_affinity"


# Settings for feature generation (must match the original training scripts)
ECFP_RADIUS = 6
ECFP_N_BITS = 2048


class BindingAffinityScorer:
    """
    Port of `Final_Molecule_Gen/ScoringFunction/BindingAffinityScorer.py`.

    Predicts binding affinity for NLRP3 using a pre-trained XGBoost regression model.
    """

    def __init__(self, assets_dir: str | Path | None = None):
        assets = Path(assets_dir) if assets_dir is not None else _assets_dir()

        # 1) Import KRFingerprints with the same API as before
        try:
            from objectives.kr_fingerprints import KRFingerprints
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "KRFingerprints implementation not available (expected in objectives/kr_fingerprints.py)"
            ) from e

        self._KRFingerprints = KRFingerprints
        self.assets_dir = assets

        # Artifacts from feature generation / reduction
        reduced_features_file = assets / "train_features_reduced.csv"
        mordred_scaler_path = assets / "mordred_scaler.pkl"

        # Artifacts from Lasso feature selection
        lasso_dir = assets / "lasso_baseline"
        imputer_path = lasso_dir / "lasso_imputer.pkl"
        scaler_path = lasso_dir / "lasso_scaler.pkl"

        # Artifacts from final XGBoost training
        xgb_dir = assets / "xgb_final"
        final_features_path = xgb_dir / "final_selected_features.json"
        final_model_path = xgb_dir / "xgb_model_on_lasso_features.ubj"

        required = [
            reduced_features_file,
            mordred_scaler_path,
            imputer_path,
            scaler_path,
            final_features_path,
            final_model_path,
            assets / "krfp.json",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            missing_list = "\n".join(f"- {p}" for p in missing)
            raise FileNotFoundError(
                "BindingAffinityScorer assets are missing.\n"
                "Expected the following files under `Molecule_Generation_Workflow/objectives/assets/binding_affinity/`:\n"
                f"{missing_list}\n\n"
                "If these artifacts are not checked into git, place them in that folder before running."
            )

        try:
            self.reduced_feature_cols = (
                pd.read_csv(reduced_features_file)
                .drop(columns=["zinc_id", "smiles", "⟨ΔΗ⟩"])
                .columns
            )
            self.mordred_scaler = joblib.load(mordred_scaler_path)
            self.imputer = joblib.load(imputer_path)
            self.scaler = joblib.load(scaler_path)
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(final_model_path))
            with open(final_features_path, "r") as f:
                self.final_236_features = json.load(f)
            self.running_avg_reward = None
            self.running_std_reward = None
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Error loading binding-affinity assets from {assets}: {e}") from e

    def generate_all_features(self, dataframe: pd.DataFrame):
        df_krfp = self._KRFingerprints.GenerateKRFingerprintsToDataFrame(
            dataframe, structures_column="smiles", verbose=False
        )

        ecfp_features = [
            list(
                AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi), ECFP_RADIUS, nBits=ECFP_N_BITS
                )
            )
            if Chem.MolFromSmiles(smi)
            else [0] * ECFP_N_BITS
            for smi in dataframe["smiles"]
        ]
        df_ecfp = pd.DataFrame(ecfp_features, columns=[f"ECFP_{i}" for i in range(ECFP_N_BITS)])

        calc = Calculator(descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(smi) for smi in dataframe["smiles"]]
        df_mordred = calc.pandas(mols, quiet=True).apply(pd.to_numeric, errors="coerce")

        return pd.concat([df_krfp.reset_index(drop=True), df_ecfp, df_mordred], axis=1)

    async def generate_all_features_async(self, dataframe: pd.DataFrame):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, self.generate_all_features, dataframe)
            return result

    async def predict(self, smiles: list[str]) -> np.ndarray:
        """
        Return predictions (higher is better) aligned with the input SMILES list.

        Note: the original code negated the model output (`predictions * -1`), preserved here.
        """
        if not hasattr(np, "product"):
            np.product = np.prod

        smile_df = pd.DataFrame(data={"smiles": smiles})
        raw_features_df = await self.generate_all_features_async(smile_df)

        expected_mordred_cols = self.mordred_scaler.feature_names_in_
        mordred_df_aligned = raw_features_df.reindex(columns=expected_mordred_cols, fill_value=0)
        raw_features_df[expected_mordred_cols] = self.mordred_scaler.transform(mordred_df_aligned)

        features_reduced = raw_features_df[self.reduced_feature_cols]
        features_imputed = self.imputer.transform(features_reduced)
        features_scaled = self.scaler.transform(features_imputed)

        features_scaled_df = pd.DataFrame(features_scaled, columns=features_reduced.columns)
        final_prepared_data = features_scaled_df[self.final_236_features]

        preds = self.model.predict(final_prepared_data)
        preds = preds * -1

        # Reduce memory churn in long RL runs
        del raw_features_df, features_scaled_df, final_prepared_data
        gc.collect()

        return preds


def predict_sync(scorer: BindingAffinityScorer, smiles: list[str]) -> list[float]:
    """
    Sync helper for `reward.compute_rewards` (which is sync).
    """
    preds = asyncio.run(scorer.predict(smiles))
    return [float(x) for x in preds]

