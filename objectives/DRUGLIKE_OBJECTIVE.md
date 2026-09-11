# Druglike multi-objective

Living spec for the simple druglike REINFORCE objective (`MGW_OBJECTIVE=druglike`), intended for chemical-tether efficacy demos (Mode I ablation / Mode II D* shells) without CO2’s peaked amine reward or NLRP BA.

## Formula

Unbounded additive physchem (no peak / unit-interval ceilings on the terms):

```text
raw = 0.35·logit(QED) + 0.25·(−SCScore) + 0.15·MolLogP + 0.25·(−log freq)
```

| Term | Definition |
|------|------------|
| logit(QED) | `log(q / (1−q))` with `q` clipped to `[1e-6, 1−1e-6]` — unbounded as QED → 1 |
| −SCScore | raw SCScore (~[1, 5], higher=harder) negated; **not** mapped/clipped to [0, 1] |
| MolLogP | raw `Descriptors.MolLogP` — unbounded both ways (no 2.5 peak) |
| Diversity | per gated mol: `−log(count(scaffold) / n_gated)`; acyclic → canonical SMILES as key |

**Hard gate:** RDKit parse + `atom_check`. Fail → `gate_ok=False`, reward `INVALID_PENALTY=-1`. No shape exclusion, BA, or PFC.

**Policy signal:** z-score of `raw` among gated molecules in the batch; ungated keep `-1`. Scale of `raw` is free because of this z-score.

## RewardBatch slot map

| Field | Druglike meaning |
|-------|------------------|
| `raw_ba` | logit(QED) |
| `raw_div` | −log scaffold freq |
| `raw_physchem` | −SCScore |
| `mw_scale` | MolLogP |
| `quality` | raw_total (gated) |
| `raw_total` | weighted sum above |

## How to run

Mode I tether ablation (none / kl_only / chem_tether / current):

```bash
bash run_druglike_tether_ablation.sh
```

Mode II D* shells (default out `outputs/druglike_dstar_arms_nn_c4`):

```bash
bash run_hcmt_dstar_arms.sh
```

Self-check:

```bash
MGW_OBJECTIVE=druglike PYTHONPATH=. uv run python objectives/check_druglike_reward.py
```

Code: [`druglike_reward.py`](druglike_reward.py), wired in [`../reward.py`](../reward.py).
