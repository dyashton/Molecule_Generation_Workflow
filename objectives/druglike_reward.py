"""Simple druglike multi-objective for tether efficacy demos.

Unbounded additive physchem (no unit-interval or peak ceilings):

    raw = 0.35·logit(QED) + 0.25·(−SCScore) + 0.15·MolLogP + 0.25·(−log freq)

logit(QED) and −log(batch scaffold freq) are unbounded above; MolLogP and
−SCScore are unbounded in at least one direction. Policy still batch-z-scores
``raw`` among gated mols.

Hard gate: RDKit parse + atom_check only (no shape exclusion / BA / PFC).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from objectives.co2_rl_helpers import murcko_scaffold_smiles
from objectives.synthesizability import SynthesizabilityScorer
from transpharmer.SMILEValidifier import atom_check

INVALID_PENALTY = -1.0

W_QED = 0.35
W_SA = 0.25
W_LOGP = 0.15
W_DIV = 0.25

# logit clip — ponytail: QED=0/1 would be ±inf; 1e-6 is plenty for ranking.
QED_EPS = 1e-6


@dataclass(frozen=True)
class DruglikeBatch:
    policy: list[float]
    raw_total: list[float]
    qed: list[float]
    # Logged name kept for RewardBatch slot map; value is −SCScore.
    sa_good: list[float]
    # Logged name kept for slot map; value is raw MolLogP.
    logp_soft: list[float]
    diversity: list[float]
    gate_ok: list[bool]
    # Pre-zscore quality for elite-style logging (= raw_total when gated).
    quality: list[float]


def sa_term(scscore: float) -> float:
    """Higher better: −SCScore (typically ~[−5, −1], no clip)."""
    return -float(scscore)


def qed_term(qed: float, *, eps: float = QED_EPS) -> float:
    """Higher better: logit(QED), unbounded as QED → 1."""
    q = min(max(float(qed), float(eps)), 1.0 - float(eps))
    return float(np.log(q / (1.0 - q)))


def logp_term(logp: float, **_kwargs) -> float:
    """Higher better: raw MolLogP (unbounded both ways)."""
    return float(logp)


def murcko_rarity_scores(mols: list) -> list[float]:
    """Per-mol −log(scaffold freq) in this batch. Unique → log(n), all-same → 0."""
    if not mols:
        return []
    keys = [_scaffold_key(m) for m in mols]
    counts = Counter(keys)
    n = float(len(mols))
    return [-float(np.log(counts[k] / n)) for k in keys]


# Back-compat aliases used by older checks / imports.
def sa_good_from_scscore(scscore: float) -> float:
    return sa_term(scscore)


def logp_soft_score(logp: float, **kwargs) -> float:
    return logp_term(logp, **kwargs)


def _scaffold_key(mol) -> str:
    """Murcko SMILES, or canonical molecule SMILES if acyclic / unavailable."""
    scaf = murcko_scaffold_smiles(mol)
    if scaf is not None:
        return scaf
    return Chem.MolToSmiles(mol)


def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + 1e-8)


class DruglikeObjectiveScorer:
    """Parse+atom_check gate; unbounded additive logit-QED / −SCScore / logP / rarity."""

    def __init__(self, sa_scorer: SynthesizabilityScorer | None = None):
        self._sa = sa_scorer

    def _get_sa(self) -> SynthesizabilityScorer:
        if self._sa is None:
            self._sa = SynthesizabilityScorer()
        return self._sa

    def compute_batch(self, smiles_list: list[str]) -> DruglikeBatch:
        n = len(smiles_list)
        if n == 0:
            return DruglikeBatch([], [], [], [], [], [], [], [])

        qed_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        sa_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        logp_arr = np.full(n, INVALID_PENALTY, dtype=np.float64)
        diversity = np.full(n, INVALID_PENALTY, dtype=np.float64)
        gate_ok = [False] * n
        raw = np.full(n, INVALID_PENALTY, dtype=np.float64)
        quality = np.full(n, INVALID_PENALTY, dtype=np.float64)
        gated_mols: list = []
        gated_idx: list[int] = []

        sa_scorer = self._get_sa()
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not atom_check(smi):
                continue
            try:
                q = float(QED.qed(mol))
                sc = float(sa_scorer.score(smi))
                lp = float(Descriptors.MolLogP(mol))
            except Exception:
                continue
            gate_ok[i] = True
            qed_arr[i] = qed_term(q)
            sa_arr[i] = sa_term(sc)
            logp_arr[i] = logp_term(lp)
            gated_mols.append(mol)
            gated_idx.append(i)

        rarities = murcko_rarity_scores(gated_mols)
        for j, i in enumerate(gated_idx):
            div = float(rarities[j])
            diversity[i] = div
            total = (
                W_QED * float(qed_arr[i])
                + W_SA * float(sa_arr[i])
                + W_LOGP * float(logp_arr[i])
                + W_DIV * div
            )
            raw[i] = total
            quality[i] = total

        # Policy: z-score raw among gated only; ungated keep INVALID_PENALTY.
        policy = np.full(n, INVALID_PENALTY, dtype=np.float64)
        if gated_idx:
            gated_raw = raw[gated_idx]
            z = _zscore(gated_raw)
            for j, i in enumerate(gated_idx):
                policy[i] = float(z[j])

        return DruglikeBatch(
            policy=[float(x) for x in policy.tolist()],
            raw_total=[float(x) for x in raw.tolist()],
            qed=[float(x) for x in qed_arr.tolist()],
            sa_good=[float(x) for x in sa_arr.tolist()],
            logp_soft=[float(x) for x in logp_arr.tolist()],
            diversity=[float(x) for x in diversity.tolist()],
            gate_ok=gate_ok,
            quality=[float(x) for x in quality.tolist()],
        )


_SCORER: DruglikeObjectiveScorer | None = None


def get_druglike_scorer() -> DruglikeObjectiveScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = DruglikeObjectiveScorer()
    return _SCORER
