# CO₂ binding RL reward contract (HPC)

Frozen for closed-loop discovery until the BE_GM surrogate passes gates.

## Objective

```bash
export MGW_OBJECTIVE=co2_binding_surrogate
export MGW_BINDING_PACK=${MGW_BINDING_PACK:-$PWD/artifacts/binding_predictor/pack.joblib}
export MGW_BINDING_HEAD=ie
export MGW_BINDING_STRONG_BONUS=${MGW_BINDING_STRONG_BONUS:-0.5}
```

| Knob | Value | Why |
|------|-------|-----|
| `MGW_BINDING_HEAD` | `ie` | Pack `reward_head` falls back to IE; BE_GM compose Spearman is below the 0.6 gate |
| Strong bonus | optional `0.5 * P(BE_GM < -6)` | Tail emphasis without trusting BE_GM regression |
| OOD gate | Morgan Tanimoto floor 0.35 **or** Mahalanobis > train 95th | Trees go quiet OOD |
| Uncertainty gate | default `2 ×` active-head OOF RMSE | Drop high-variance scores from the policy |

## Reward

For gated SMILES:

```text
reward = -binding_hat + strong_bonus * P(strong)
         - inorganic_penalty - charged_penalty - mw_penalty
```

`binding_hat` is **IE** under this contract. Failures / OOD → `NaN` (masked from REINFORCE).

## Artifacts

Required under `artifacts/binding_predictor/`:

- `pack.joblib`, `ie.joblib`, `be.joblib`, `penalty.joblib`, `strong.joblib`
- optional: `metrics.json`, `oof.csv`, `begm_direct.joblib`

Fetch with `scripts/fetch_binding_pack.sh` (sibling monorepo copy or release tarball).

## Do not claim

- BE_GM-optimized RL while `MGW_BINDING_HEAD=ie`
- DFT as the inner-loop reward (DFT is the Phase-2 oracle for AL/BO only)
