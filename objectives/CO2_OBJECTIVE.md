# CO2 objective, penalties, and RL drivers

Living spec for the amine CO₂-capture REINFORCE run (`MGW_OBJECTIVE=co2`).
**Source of truth is the code.** If this file and the constants disagree, the code wins.
Update this file whenever you change a weight, excess, or training driver.

| Layer | File |
|-------|------|
| Reward / physchem / penalties | [`co2_reward.py`](co2_reward.py) |
| Counts, excesses, SMARTS | [`co2_rl_helpers.py`](co2_rl_helpers.py) |
| Policy loss, KL, entropy, schedules | [`../train.py`](../train.py) |
| Default knobs | [`../train.yaml`](../train.yaml) |
| Viewer breakdown | [`../notebooks/view_top_molecules.ipynb`](../notebooks/view_top_molecules.ipynb) |

Restart from `weights/guacamol_nc.pt` after any reward change.

---

## Reward (gated molecule)

Hard gate: formal charge 0, atoms in `{C,H,N,O,F,S}` only, no sulfonyl `[SX4](=O)(=O)`, one fragment (no salts).
Failed gate → NaN (no gradient). Soft penalties never ungated a molecule.

Default path is **z-score** (tanh-capped at `Z_SOFT = 3`). Optional:

- `MGW_OBJECTIVE_CEILINGS=1` — clamp each term to `RANGES` before z-score
- `MGW_CO2_SIGMOID=1` — weighted mean of sigmoid unit terms instead of unbounded z-sum
- `MGW_INCLUDE_PFC=0` — skip Polar Focal Concavity
- `MGW_NO_PENALTIES=1` — skip additive penalties except molecular weight, and skip bonuses (physchem + diversity − MW penalty; hard gate and prior-FP floor still apply)

```text
physchem = W_APOL·z(apol/n_HA)
         + W_TPSA·z(TPSA/n_HA)
         + W_BASICITY·z(basicity/n_HA)
         + W_LOGP·z(LogP)·(−1)
         + W_HALL_KIER_ALPHA·z(HallKierAlpha)
         + W_PFC_COMPOSITE·z(PFC)          # 1.25× if PFC on
reward   = physchem + W_DIVERSE·z(diversity) − MW penalty − Σ other W_i · excess_i + bonuses
quality  = reward  (after prior-FP floor)
policy   = quality × prior_fp_factor
```

`n_HA = max(heavy atoms, 1)`. Per-atom scaling is on **apol / TopoPSA / Mordred basicity** only (extensive). LogP, HallKierAlpha, PFC are already intensive. The old HAE multiplier is **not** applied.

All `W_*` physchem/diversity weights are `1.0` except PFC `1.25`.

`REF_STATS` / `_BASE_RANGES` for the intensive terms are GuacaMol-prior gated mean/std and GuacaMol∪AxelRolov p10/p90 of `col/n_HA`, then `RANGE_PAD = 0.05`. Recalibrate with `PYTHONPATH=. uv run python objectives/calibrate_co2_ranges.py`.

Aliphatic amines (`count_amine_n`) are 1°/2°/3° N on carbon, not amides or aromatic N.

---

## Additive penalties (`reward −= W × excess`)

Uncapped unless noted. Default `W = 1.0` except MW (5.0), OH (2.0), alkyne (2.0).

| Name | Excess | W | Notes |
|------|--------|---|-------|
| MW | 0 on 100–350 Da; ramps to 1 at 50 / 400; continues outside | 5.0 | `mw_scale` is logged, not multiplied |
| SCScore | 0 at ≤3.5; 1 at 5; linear past 5 | 1.0 | |
| Max consecutive rotatable bonds | 0 at ≤4; 1 at 7; linear past 7 | 1.0 | arm length, not total rot |
| Total rotatable bonds | `max(0, 3−n) + max(0, n−6)` | 1.0 | Lipinski `NumRotatableBonds`; ideal 3–6 |
| Amine-justified OH | `max(0, n_OH − n_amine)` | 2.0 | 1:1 alkanolamines free; polyols grow |
| Amine floor | `max(0, 3 − n_amine) / 3` | 1.0 | 0 amines → 1; ≥3 → 0 |
| N···N distance | `max(0, 2.8 − d_min) / 2.8` | 1.0 | graph dist ≥4; NaN → 0 |
| Epoxide / aziridine | match count `[N,O;r3]` | 1.0 | |
| Sulfur | S-atom count (free = 0) | 1.0 | |
| Ester | `C(=O)[OX2H0][#6]` count (not acids) | 1.0 | |
| Carboxylic acid | `[CX3](=O)[OX2H1]` count | 1.0 | disjoint from ester |
| Cyclic acetal | dioxolane + dioxane matches minus 1 free | 1.0 | |
| Aminal | `[CX4]([NX3])([NX3])` | 1.0 | geminal diamine |
| Hydrate | `[CX4]([OX2H])([OX2H])` | 1.0 | geminal diol |
| Hemiaminal | `[CX4]([NX3])([OX2])` | 1.0 | geminal N+O |
| Peroxide | `[OX2][OX2]` count | 1.0 | any O–O |
| Peracid | `[CX3](=O)[OX2][OX2]` count | 1.0 | also fires peroxide (−2 total) |
| Amide | `[NX3][CX3](=O)` count | 1.0 | also in non-basic N (−2 total) |
| Hindered amine α | `[NX3;!$(NC=O)][CX4]([#6])([#6])` unique N–α-C | 1.0 | iPr / tBu / cyclohexyl-type |
| Ring oxygen | `max(0, n_ring_O − 1)` | 1.0 | morpholine/furan free; sugars grow |
| Alkyne | `[$([CX2]#C)]` count | 2.0 | C≡C only, not nitrile |
| Aromatic furan | `o1cccc1` count | 1.0 | not THF |
| Non-basic N | `max(0, n_N − n_amine)` | 1.0 | azoles, amides, anilines |

---

## Additive bonuses (`reward += W × units`)

| Name | Units | W | Notes |
|------|-------|---|-------|
| Azacrown | `min(2, max(0, max_ring_aliphatic_N − 2))` | 1.0 | piperazine 0; TACN 1; cyclen 2 |
| Vicinal diamine-diol | 1 if primary 1,2-diamine has OH or CH2OH on both carbons | 2.0 | EDA/MEA 0 |

---

## Multiplicative drivers (after additive reward)

| Name | Effect | Default (`train.yaml`) |
|------|--------|------------------------|
| Scaffold DF | **Off** (`SCAFFOLD_BUCKET=0`). Helper still exists; not applied. | was bucket 40 × 0.5 |
| Prior-FP tether | Morgan Tanimoto vs cached prior bank; below floor 0.3, scale linearly | `PRIOR_FP_FLOOR=0.3`, bank 2048 |

Independent of the loss-side chemical tether. Leave this in place for later KL vs prior-FP vs chemical-tether comparisons (`PRIOR_FP_FLOOR: 0` disables).

---

## Policy loss (training, not reward)

```text
L = −mean(log π(x) · A) − λ_H · H + λ_KL · KL(π ‖ π_ref) + λ_tether · mean(log π · (c − c̄))
```

`A` is advantage vs EMA baseline (`BASELINE_MOMENTUM=0.9`), optionally restricted to the top quantile of gated rewards.

`d` is the hierarchical chemical-tether distance (detached SMILES feature, not backprop through fingerprints/embeddings). Each channel is divided by the bank’s median leave-one-out k-NN (`s_c`, stored in the `.npz`) so `λ=1` is an equal vote in typical-NN units (on-manifold `d ≈ 1`). Failed encodes are dropped and `λ` is renormalized for that molecule.

```text
d'_c = d_c / s_c
d_i  = (λ_m d'_Morgan + λ_2d d'_2D + λ_3d d'_3D) / (λ_m + λ_2d + λ_3d)
```

- Morgan: `1 −` Tanimoto to k-NN on the GuacaMol∪AxelRolov bank (`CHEM_TETHER_K`, default 1)
- 2D: cosine distance to ChemBERTa (`DeepChem/ChemBERTa-77M-MLM`) k-NN
- 3D: cosine distance to Uni-Mol k-NN

Do not average raw cosine with Tanimoto; `s_c` is the unit conversion. `λ` is not `1/s_c`. Explore `D*` is on this calibrated scale (`1` = typical bank neighbor).

Cost `c` depends on `CHEM_TETHER_MODE` (default **`regularize`**):

| Mode | Cost | Meaning |
|------|------|---------|
| `regularize` | `c = d` | Mode I: stay near the chemical manifold |
| `explore` | `c = (d − D*)²` | Mode II: novelty shell around radius `D*` |

When `MODE=explore`, `D*` is controlled by `CHEM_TETHER_D_STAR_SCHEDULE`:

| Schedule | `D*` |
|----------|------|
| `static` (default) | fixed `CHEM_TETHER_D_STAR` (default 1.0, NN-units) |
| `adaptive` | linear `CHEM_TETHER_D_STAR_START` → `END` over `EXPLORE_EPISODES` |

Explore runs should set `PRIOR_FP_FLOOR: 0` so the Morgan reward floor does not fight the novelty shell. Matched D* arms (`run_hcmt_dstar_arms.sh`) use fixed `λ_tether=0.75`, fixed `KL=0.50`, and `D* ∈ {0.80, 1.00, 1.30, 2.00}` (NN-units) under `outputs/druglike_dstar_arms_nn/`.

Episode CSV logs `tether_d` (batch mean calibrated `d`), per-channel raw `tether_d_*` and NN-unit `tether_dp_*`, `tether_d_star`, `tether_abs_dev` = mean `|d−D*|`, and `tether_shell_hit` = fraction with `|d−D*| ≤ 0.15` (`SHELL_HIT_BAND`, explore mode).

Bank: `weights/chem_tether_bank.npz` from `objectives/build_chemical_tether_bank.py`.

| Driver | Role | Schedule (`train.yaml`) |
|--------|------|-------------------------|
| Temperature | sampling | 1.2 → 0.75 over `EXPLORE_EPISODES=1000` |
| Entropy coef `λ_H` | keep token entropy up | 0.08 → 0.04 |
| KL coef `λ_KL` | stay near frozen GuacaMol prior | 0.75 → 0.40 |
| Chemical tether `λ_tether` | Mode I minimize `d` / Mode II shell at `D*` | coef 0.75 → 0.40; mode `regularize` |
| Top quantile | fraction of gated batch in the policy gradient | 1.0 → 0.5 |
| Inception replay | mix elite SMILES into the batch | k=4, boost to 16 after 25 decline streaks, max 50% of batch |
| Min gated | skip thin batches | 8 |
| Grad clip | | 0.5 |

Three independent regularizers (do not delete any of them):

| Regularizer | Where | Disable |
|-------------|--------|---------|
| KL(π ‖ π_ref) | policy loss | `KL_COEF_START/END: 0` |
| Morgan prior-FP floor | reward multiply | `PRIOR_FP_FLOOR: 0` |
| Chemical tether | policy loss | `CHEM_TETHER_COEF*: 0` |

Episode CSV `rl_metrics.csv` always has `kl`, `prior_fp_mean_factor`, `tether_d`, `tether_d_*`, `tether_d_star` (plus explore shell diagnostics). `CHEM_TETHER_LOG: true` records manifold distances on eval/snapshots without applying the tether loss.

Later comparison (commented in `train.yaml`): current stack / KL-only / prior-FP-only / chemical-tether-only.

Decline restore / gate-frac restore / early stop are training guards, not reward terms (`DECLINE_PATIENCE=200`, `GATE_FRAC_FLOOR=0.20`, `EARLY_STOP_PATIENCE=400`).

---

## Not in the score

- HAE multiplier (`clip(1 + 0.35·z_HAE, 0.25, 3)`) — removed; `/n_HA` is on the descriptors
- `n_basic / n_HA` as a bonus
- Hard HA / amine count gates (they collapsed pass rate)
- PAINS catalog
