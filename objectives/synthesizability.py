from __future__ import annotations

from pathlib import Path


def _here() -> Path:
    return Path(__file__).resolve().parent


class SynthesizabilityScorer:
    """
    Wrapper around the SCScore model used in `Final_Molecule_Gen/SAScoringFunction/SAScorer.py`.

    The model artifact path is repo-relative so the workflow remains relocatable.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
    ):
        from objectives.scscore_numpy import SCScorer

        self.scorer = SCScorer()
        default = (
            _here()
            / "assets"
            / "sa_model"
            / "full_reaxys_model_1024bool"
            / "model.ckpt-10654.as_numpy.json.gz"
        )
        p = Path(model_path) if model_path is not None else default
        self.model_path = p
        if not p.exists():
            raise FileNotFoundError(
                "Synthesizability model file not found.\n"
                "Expected SCScore weights at:\n"
                f"- {p}\n\n"
                "Place `model.ckpt-10654.as_numpy.json.gz` under "
                "`Molecule_Generation_Workflow/objectives/assets/sa_model/full_reaxys_model_1024bool/` "
                "or pass `model_path=` explicitly."
            )
        self.scorer.restore(str(p))

    def score(self, smiles: str) -> float:
        (_smi, score) = self.scorer.get_score_from_smi(smiles)
        return float(score)

