#!/usr/bin/env python3
"""Pool MGW CO2 molecules and re-rank by literature affinity checklist.

Affinity-first: does not hard-filter MW / SCScore / esters / amides.
Hard-rejects only chemically unstable motifs. Tie-break with raw_physchem.

Usage (from Molecule_Generation_Workflow/):
  PYTHONPATH=. uv run python scripts/rerank_co2_literature_top5.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops

from objectives.co2_rl_helpers import (
    _epoxide_aziridine_smarts,
    _hindered_amine_alpha_smarts,
    _peracid_smarts,
    _peroxide_smarts,
    count_amine_n,
    count_oh,
    count_smarts_matches,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "outputs" / "output_1"
DEFAULT_OUT_DIR = ROOT / "presentation_assets" / "co2"

POOL_SIZE = 2000
TOP_N = 5
TOP_AUDIT = 50
TANIMOTO_MAX = 0.7
AMINE_TERM_CAP = 6
MULTI_N_CAP = 4
OH_GRAPH_DIST = 4

# 1° / 2° aliphatic amine (excludes amide via CX4 neighbor requirement).
_PRIMARY_AMINE = Chem.MolFromSmarts("[NX3H2][CX4]")
_SECONDARY_AMINE = Chem.MolFromSmarts("[NX3H]([CX4])[CX4]")
_GUANIDINE = Chem.MolFromSmarts("N=C(N)N")
_AMIDINE = Chem.MolFromSmarts("N=C(N)C")
_ALIPHATIC_OH = Chem.MolFromSmarts("[CX4][OX2H]")

# Hard-reject liabilities not already covered by co2_rl_helpers.
_HARD_REJECT_SMARTS: list[tuple[str, Chem.Mol]] = [
    ("peroxide", _peroxide_smarts()),
    ("peracid", _peracid_smarts()),
    ("epoxide_aziridine", _epoxide_aziridine_smarts()),
    ("azide", Chem.MolFromSmarts("[N-]=[N+]=N")),
    ("diazo", Chem.MolFromSmarts("C=[N+]=[N-]")),
    ("isocyanate", Chem.MolFromSmarts("N=C=O")),
    ("isothiocyanate", Chem.MolFromSmarts("N=C=S")),
    ("acyl_halide", Chem.MolFromSmarts("C(=O)[F,Cl,Br,I]")),
    ("sulfonyl_halide", Chem.MolFromSmarts("S(=O)(=O)[F,Cl,Br,I]")),
    ("phosphoryl_halide", Chem.MolFromSmarts("P(=O)[F,Cl,Br,I]")),
    ("hetero_halogen", Chem.MolFromSmarts("[N,O,S][F,Cl,Br,I]")),
    ("nitrate_ester", Chem.MolFromSmarts("O=[N+]([O-])O[#6]")),
    ("nitrite_ester", Chem.MolFromSmarts("O=NO[#6]")),
    ("nitramine", Chem.MolFromSmarts("[NX3][N+](=O)[O-]")),
]


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _safe_float(v, default: float = float("-inf")) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN
        return default
    return x


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def hard_reject_reason(mol: Chem.Mol) -> str | None:
    if mol is None:
        return "sanitize_fail"
    if Chem.GetFormalCharge(mol) != 0:
        return "formal_charge"
    if len(Chem.GetMolFrags(mol)) != 1:
        return "multi_fragment"
    allowed = {1, 6, 7, 8, 9, 16}  # H,C,N,O,F,S
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in allowed:
            return f"forbidden_atom_{atom.GetSymbol()}"
    for name, smarts in _HARD_REJECT_SMARTS:
        if smarts is None:
            continue
        if count_smarts_matches(mol, smarts) > 0:
            return name
    return None


def _amine_n_indices(mol: Chem.Mol) -> set[int]:
    """Atom indices of aliphatic 1°/2°/3° amine N (same family as count_amine_n)."""
    idxs: set[int] = set()
    for smarts in (_PRIMARY_AMINE, _SECONDARY_AMINE):
        if smarts is None:
            continue
        for match in mol.GetSubstructMatches(smarts):
            idxs.add(match[0])
    # Tertiary aliphatic amines from helper SMARTS family.
    tert = Chem.MolFromSmarts("[NX3]([CX4])([CX4])[CX4]")
    if tert is not None:
        for match in mol.GetSubstructMatches(tert):
            idxs.add(match[0])
    return idxs


def count_unhindered_primary_secondary(mol: Chem.Mol) -> int:
    """1°/2° aliphatic amine N whose matched α-C is not doubly carbon-substituted."""
    hindered = set()
    hs = _hindered_amine_alpha_smarts()
    if hs is not None:
        for match in mol.GetSubstructMatches(hs):
            hindered.add(match[0])
    n = 0
    for smarts in (_PRIMARY_AMINE, _SECONDARY_AMINE):
        if smarts is None:
            continue
        for match in mol.GetSubstructMatches(smarts):
            if match[0] not in hindered:
                n += 1
    return n


def has_amino_alcohol(mol: Chem.Mol) -> bool:
    """True if any aliphatic OH is within OH_GRAPH_DIST of an aliphatic amine N."""
    amine_idxs = _amine_n_indices(mol)
    if not amine_idxs or _ALIPHATIC_OH is None:
        return False
    oh_matches = mol.GetSubstructMatches(_ALIPHATIC_OH)
    if not oh_matches:
        return False
    # O is second atom in [CX4][OX2H]
    oh_o = {m[1] for m in oh_matches}
    dist = rdmolops.GetDistanceMatrix(mol)
    for o in oh_o:
        for n in amine_idxs:
            if dist[o][n] <= OH_GRAPH_DIST:
                return True
    return False


def affinity_score(mol: Chem.Mol) -> tuple[float, dict]:
    """Return (score, motif breakdown). Higher = better affinity proxy."""
    n_unh = count_unhindered_primary_secondary(mol)
    n_amine = count_amine_n(mol)
    has_g = count_smarts_matches(mol, _GUANIDINE) > 0
    has_a = count_smarts_matches(mol, _AMIDINE) > 0
    amino_oh = has_amino_alcohol(mol)

    amine_term = min(AMINE_TERM_CAP, 2 * n_unh)
    multi_n = min(MULTI_N_CAP, max(0, n_amine - 1))
    superbase = 3 if (has_g or has_a) else 0
    hbd = 1 if amino_oh else 0
    score = float(amine_term + multi_n + superbase + hbd)

    motifs = {
        "n_unhindered_1_2_amine": n_unh,
        "n_amine": n_amine,
        "n_oh": count_oh(mol),
        "guanidine": int(has_g),
        "amidine": int(has_a),
        "amino_alcohol": int(amino_oh),
        "amine_term": amine_term,
        "multi_n_term": multi_n,
        "superbase_term": superbase,
        "hbd_term": hbd,
    }
    return score, motifs


def morgan_fp(mol: Chem.Mol):
    try:
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprint(mol)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _consider(best: dict[str, dict], smi: str, phys: float, policy: float, source: str) -> None:
    """Keep best physchem per canonical SMILES; skip hard-reject motifs early."""
    canon = canonicalize(smi)
    if canon is None:
        return
    mol = Chem.MolFromSmiles(canon)
    if hard_reject_reason(mol) is not None:
        return
    key = (phys, policy)
    prev = best.get(canon)
    if prev is None or key > (prev["raw_physchem"], prev["policy"]):
        best[canon] = {
            "SMILES": canon,
            "raw_physchem": phys,
            "policy": policy,
            "source": source,
        }


def collect_pool(outputs_dir: Path, pool_size: int) -> list[dict]:
    """Best policy/physchem per canonical SMILES from all *_co2_* runs.

    Hard-reject (I-hacks, peroxides, …) before taking the top-N so iodine
    reward exploits do not crowd out clean polyamines.
    """
    best: dict[str, dict] = {}
    run_dirs = sorted(
        p for p in outputs_dir.iterdir() if p.is_dir() and "_co2_" in p.name
    )
    for run_dir in run_dirs:
        rel_run = str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir)

        # Snapshots
        snap_dir = run_dir / "molecule_snapshots"
        if snap_dir.is_dir():
            for csv_path in sorted(snap_dir.glob("ep*.csv")):
                with csv_path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not _parse_bool(row.get("valid", False)):
                            continue
                        if not _parse_bool(row.get("gate_ok", False)):
                            continue
                        smi = (row.get("smiles") or "").strip()
                        if not smi:
                            continue
                        _consider(
                            best,
                            smi,
                            _safe_float(row.get("raw_physchem")),
                            _safe_float(row.get("policy")),
                            f"{rel_run}/molecule_snapshots/{csv_path.name}",
                        )

        # analysis*/data/top_hits.csv
        for analysis in sorted(run_dir.glob("analysis*")):
            hits = analysis / "data" / "top_hits.csv"
            if not hits.is_file():
                continue
            with hits.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smi = (row.get("SMILES") or row.get("smiles") or "").strip()
                    if not smi:
                        continue
                    canon = canonicalize(smi)
                    if canon is None:
                        continue
                    prev = best.get(canon)
                    # Prefer snapshot physchem when already present.
                    if prev is not None and "top_hits" not in prev["source"]:
                        continue
                    reward = _safe_float(row.get("reward"))
                    _consider(
                        best,
                        smi,
                        reward,
                        reward,
                        f"{rel_run}/{analysis.name}/data/top_hits.csv",
                    )

    ranked = sorted(
        best.values(),
        key=lambda r: (r["raw_physchem"], r["policy"]),
        reverse=True,
    )
    return ranked[:pool_size]


def diversify(rows: list[dict], k: int, tanimoto_max: float) -> list[dict]:
    selected: list[dict] = []
    fps = []
    for row in rows:
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        fp = morgan_fp(mol)
        if any(DataStructs.TanimotoSimilarity(fp, sfp) >= tanimoto_max for sfp in fps):
            continue
        selected.append(row)
        fps.append(fp)
        if len(selected) >= k:
            break
    return selected


def score_pool(pool: list[dict]) -> tuple[list[dict], dict]:
    scored: list[dict] = []
    reject_counts: dict[str, int] = defaultdict(int)
    for row in pool:
        mol = Chem.MolFromSmiles(row["SMILES"])
        reason = hard_reject_reason(mol)
        if reason is not None:
            reject_counts[reason] += 1
            continue
        score, motifs = affinity_score(mol)
        motif_str = (
            f"unh1_2={motifs['n_unhindered_1_2_amine']};"
            f"amine={motifs['n_amine']};"
            f"oh={motifs['n_oh']};"
            f"guan={motifs['guanidine']};"
            f"amid={motifs['amidine']};"
            f"amino_oh={motifs['amino_alcohol']}"
        )
        scored.append(
            {
                **row,
                "affinity_score": score,
                "n_amine": motifs["n_amine"],
                "motifs": motif_str,
                "reject_notes": "",
            }
        )
    scored.sort(
        key=lambda r: (r["affinity_score"], r["raw_physchem"], r["policy"]),
        reverse=True,
    )
    return scored, dict(reject_counts)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = i
            w.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--top-audit", type=int, default=TOP_AUDIT)
    ap.add_argument("--tanimoto-max", type=float, default=TANIMOTO_MAX)
    args = ap.parse_args()

    print(f"Pooling from {args.outputs_dir} …")
    pool = collect_pool(args.outputs_dir, args.pool_size)
    print(f"Pool size (unique canonical, top by physchem): {len(pool)}")

    # Pool is already hard-reject filtered; score_pool is a second pass for safety.
    scored, reject_counts = score_pool(pool)
    print(f"Scored: {len(scored)} (late rejects {sum(reject_counts.values())})")
    if reject_counts:
        print("Late reject reasons:", reject_counts)

    top5 = diversify(scored, args.top_n, args.tanimoto_max)
    if len(top5) < args.top_n:
        print(
            f"Warning: only {len(top5)} diverse molecules under "
            f"Tanimoto<{args.tanimoto_max}; filling from score order."
        )
        seen = {r["SMILES"] for r in top5}
        for row in scored:
            if row["SMILES"] in seen:
                continue
            top5.append(row)
            seen.add(row["SMILES"])
            if len(top5) >= args.top_n:
                break

    fields = [
        "rank",
        "SMILES",
        "affinity_score",
        "raw_physchem",
        "policy",
        "n_amine",
        "motifs",
        "reject_notes",
        "source",
    ]
    top5_path = args.out_dir / "co2_literature_top5.csv"
    audit_path = args.out_dir / "co2_literature_ranked.csv"
    write_csv(top5_path, top5, fields)
    write_csv(audit_path, scored[: args.top_audit], fields)
    print(f"Wrote {top5_path}")
    print(f"Wrote {audit_path} (top {min(args.top_audit, len(scored))})")
    for i, row in enumerate(top5, 1):
        print(
            f"  {i}. score={row['affinity_score']:.0f} "
            f"phys={row['raw_physchem']:.3f} amines={row['n_amine']} "
            f"{row['motifs']}  {row['SMILES'][:60]}"
        )


if __name__ == "__main__":
    main()
