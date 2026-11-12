# Molecule Generation Workflow

Self-contained **REINFORCE** fine-tuning for the vendored **transpharmer** SMILES GPT. Define objectives by editing **`reward.py`** (`compute_rewards`). Training artifacts go under **`outputs/`**; inference runs write to **`inference_outputs/`**.

## Layout

| Path | Purpose |
|------|---------|
| `reward.py` | **Edit `compute_rewards(smiles_list)`** for your reward(s). |
| `train.py` | RL training CLI. |
| `generate.py` | Sampling + scoring CLI. |
| `train.yaml` / `generate.yaml` | Device, model, generation, RL hyperparameters, output dirs. |
| `transpharmer/` | Vendored generator (GPT tokenizer + `generate_yield`). |
| `weights/guacamol_nc.pt` | Default pretrained checkpoint. |
| `outputs/checkpoints/` | Saved `.pt` files (periodic + best; see below). |
| `outputs/graphs/` | `final.png` (every episode, full training curve), `best.png` (overwritten on each new best raw objective). |
| `inference_outputs/<RUN_NAME>/` | `raw_smiles.csv`, `scored_smiles.csv`. |

## Setup (uv)

```bash
cd Molecule_Generation_Workflow
uv sync
```

Use **Python 3.11** (RDKit wheels; `uv` follows `.python-version`). Paths in YAML files are resolved **relative to the YAML file’s directory**.

## HPC: CO₂ binding RL (closed-loop generator)

Branch `co2-binding-hpc` packages the binding-surrogate RL objective for clone-on-cluster use (proposals → Bayesian AL in `CO2_IE_Dataset`).

```bash
git checkout co2-binding-hpc
bash scripts/fetch_binding_pack.sh          # or path/to/binding_predictor.tar.gz
bash scripts/run_co2_binding_generate.sh    # → artifacts/discovery/proposals.csv
# Slurm: sbatch scripts/hpc/sbatch_rl_generate.sh
```

Docs: [`docs/HPC_RL.md`](docs/HPC_RL.md), reward freeze: [`docs/REWARD_CONTRACT.md`](docs/REWARD_CONTRACT.md).

## Training

```bash
uv run python train.py --config train.yaml
# Optional: override starting weights
uv run python train.py --config train.yaml --ckpt weights/guacamol_nc.pt
```

**CO2 objective:** `MGW_OBJECTIVE=co2 uv run python train.py --config train.yaml`

Reward, penalties, and RL drivers (KL, entropy, schedules): [`objectives/CO2_OBJECTIVE.md`](objectives/CO2_OBJECTIVE.md).

**Surrogate-only CO2 IE objective:**

```bash
cd ../CO2_IE_Dataset
uv run python scripts/train_surrogate.py
cd ../Molecule_Generation_Workflow
MGW_OBJECTIVE=co2_ie_surrogate ./scripts/run_co2_ie_surrogate_rl.sh
```

**Offline DFT-weighted policy optimization:**

This path trains directly from fixed `SMILES,IE,source` rows without invoking
the DFT pipeline or a surrogate. It removes IE values outside the 1st–99th
percentile, splits by molecular scaffold, and uses bounded rank weights with a
KL anchor to the pretrained SELFIES model.

```bash
uv run python offline_train.py \
  --data ../CO2_IE_Dataset/data/compiled_ie_full100_next_dft_small300_adversarial200.csv \
  --output ./outputs/offline_dft
```

This is conservative offline policy optimization rather than trajectory-based
offline RL: the input contains terminal molecules and scalar DFT rewards, but
no behavior-policy probabilities or token-level action trajectories. The run
writes `filtered.csv`, `metrics.json`, and `checkpoints/last.pt`.

**CB gated objective** (binder classifier gate + pKi ranking):

```bash
# models already under data/CB/artifacts/
./scripts/run_cb_gated_rl.sh
```

Reward: `r = pKi + β · P(active) − λ_mw · expm1(excess / τ) − λ_div · sim_ramp − inorg − charged`
(`excess` = Da outside 350–550; `sim_ramp` is 0 below Tanimoto 0.5 and 1 for a duplicate or shared scaffold).
Defaults: `β=1`, `λ_mw=1`, `τ=50`, `λ_div=1`, inorg/charged=`1`. Entropy anneals 0.20→0.15 over 500 episodes; temperature still cools over 100.
Env: `MGW_CB_BETA`, `MGW_CB_MW_PENALTY`, `MGW_CB_MW_TAU`, `MGW_CB_DIV_PENALTY`, `MGW_CB_DIV_SIM_LO`, `MGW_CB_INORGANIC_PENALTY`, `MGW_CB_CHARGED_PENALTY`, `MGW_CB_MW_INNER_{LO,HI}`, `MGW_CB_*_MODEL`.

**M4 binding objective** (muscarinic M4 pKi, not agonism or PAM activity). The generator is the SELFIES TransPharmer prior (`weights/guacamol_nc_selfies_medium_co2.pt`, `data/mixed_selfies/vocab.json`):

```bash
# surrogates: uv run python ../data/train_muace_surrogate.py
./scripts/run_muace4_rl.sh
```

Reward terms are equalized by `1/std` on the cleaned M4 set: `0.68·pKi_M4 + 3.5·QED − 2.5·sc_penalty + w_nov·novelty + (1−batch_sim) − PAINS − inorganic + 0.96·((pKi_M4−pKi_M1)+(pKi_M4−pKi_M3)) + 4.2·CNS_MPO/6 + (1−P_pgp) + 1/(1+Cl_micro/s_micro) + (1−max CYP) + (1−P_hERG)`.
`sc_penalty` is 0 for SCScore ≤ 3 and 1 at SCScore ≥ 5. Novelty weight anneals from 0 to 1 over `N_EPISODES`. Formal charge and hepatocyte clearance are not in the reward. P-gp is inhibition (`Pgp_Broccatelli`), not efflux. The first scored batch downloads ADMET-AI weights and caches the M4-set median microsomal clearance at `data/MuAce4/artifacts/muace4_s_micro.json`. Narrative: [`objectives/M4_NARRATIVE.md`](objectives/M4_NARRATIVE.md).

**M4 docking calibration:** the optional Vina pipeline uses experimental M4
structure 5DSG and the crystallographic tiotropium site. It is post-training
validation, not an RL reward:

```bash
uv run python scripts/setup_muace4_docking.py
uv run python scripts/dock_muace4.py \
  --input ../data/MuAce4/MuAce4_pKi_clean.csv \
  --output ../data/MuAce4/docking/muace4_labeled_vina.csv \
  --center 50.893654 8.183231 63.220423
```

The docking script writes Vina scores, `vina_affinity = -vina_score`, failure
reasons, and a `.metrics.json` file with Pearson and Spearman correlation to
cleaned-set pKi. Salt forms are reduced to their largest fragment for docking.
Ligands over 60 heavy atoms are explicitly marked `skipped_too_large` because
the fixed small-molecule box is not a meaningful peptide docking benchmark.

**Surrogate-only CO2 IE notes:**

The surrogate builds an intentionally broad 10,668-feature pool from all RDKit
descriptors, aggregated multi-conformer ETKDG 3D shape descriptors, and
Morgan, MACCS, RDKit, atom-pair, and topological-torsion fingerprints. Nested
source-held-out importance selection keeps only stable features (the selected
feature list and top importances are written to the metrics JSON) before producing
`CO2_IE_Dataset/artifacts/ie_surrogate_xgb.joblib`. The surrogate RL config
starts at batch size 2048 on CPU; `DEVICE: auto` selects GPU only when the
configured batch size is 256 or smaller. The RL learning signal is
`-predicted IE - inorganic penalty`. By default, atoms outside
`C/H/N/O/F/S/Cl/Br/I` receive the configurable `INORGANIC_PENALTY` from
`train_selfies_co2_ie_surrogate.yaml`; this is a search-space preference, not
a chemistry-validity rule. Inorganic structures remain valid candidates, but
the frozen surrogate is out-of-domain for them and their predictions should
not be treated as quantitatively trustworthy. Molecules containing any
formally charged atom additionally receive `CHARGED_PENALTY`, including
net-neutral disconnected ionic pairs.

To run the `compiled_4` baseline with its compatible delta correction, use:

```bash
./scripts/run_co2_ie_compiled4_delta_rl.sh
```

This loads `CO2_IE_Dataset/artifacts/ie_surrogate_compiled_4.joblib` and adds
the correction from
`CO2_IE_Dataset/artifacts/ie_delta_surrogate_compiled_4.joblib` before applying
the inorganic and charged-molecule penalties.

To use the pooled `compiled_4.csv` neighborhood predictor directly:

```bash
./scripts/run_co2_ie_compiled4_neighborhood_rl.sh
```

This uses a similarity-weighted Morgan-fingerprint neighborhood with `k=10`,
excluding an exact canonical match from its own neighborhood query.
The neighborhood reward also applies the shared MW penalty: zero across
100–450 Da, with the high-MW exponential ramp above 450 Da and weight 5.0.

The frozen ChemBERTa benchmark is trained separately with:

```bash
cd ../CO2_IE_Dataset
uv run python scripts/train_chemberta_surrogate.py
cd ../Molecule_Generation_Workflow
./scripts/run_co2_ie_chemberta_rl.sh
```

This launcher uses the ChemBERTa artifact without replacing the RDKit/3D
artifact.

Optional CO2 knobs:

- `MGW_INCLUDE_PFC=0` — skip Polar Focal Concavity (physchem + diversity only).
- `MGW_OBJECTIVE_CEILINGS=1` — clamp raw properties to GuacaMol∪AxelRolov `RANGES` before z-scoring (limits reward hacking past either prior’s bounds). Default is unlimited.
- `MGW_CO2_SIGMOID=1` — weighted sum of `sigmoid_to_unit(raw, *RANGES)` terms (legacy CO2 scheme; LogP inverted). Default is soft-bounded z-scores.

CO2 run folders are auto-named from those flags as  
`outputs/.../YYYYMMDD_HHMMSS_co2_{full|nopf}_{z_ceil|z_uncapped|sig}/`  
(e.g. `20260727_203800_co2_full_sig`). Sigmoid runs omit ceil/uncapped because
`RANGES` already soft-limit each term.

After changing RL/reward knobs, **restart from `weights/guacamol_nc.pt`**. Do not resume an old run — reward/KL semantics changed.

Smoke check over the first ~50–100 episodes:

- Logs show `kl>0` and `learn ≈ 0.25 * n_gated` (e.g. ~12 of 48)
- EMA / gated mean trends up vs episode 1 (not stuck below start)
- No NaN `policy_loss`; `skipped_update` only on thin batches

If the mean rises then collapses, raise `RL_TRAIN.KL_COEF` (e.g. `0.1`) before touching LR.

Defaults that matter for learning: `LEARNING_RATE: 1e-5`, `TOP_QUANTILE: 0.25`, `KL_COEF: 0.05`, `ENTROPY_COEF: 0.01`.

**Checkpoints**

- **`periodic_<RUN_NAME>_ep<N>.pt`** — every `RL_TRAIN.SAVE_EVERY` episodes and on the last episode.
- **`best_<RUN_NAME>_ep<N>.pt`** — whenever mean batch reward exceeds the previous best.
- **`best.pt`** — symlink or copy of the latest best checkpoint (convenient for inference).

Each file stores `model_state_dict`, `optimizer_state_dict`, `episode`, `best_reward`, `reward_baseline`.

**Rolling cleanup:** set `RL_TRAIN.KEEP_LAST_N_PERIODIC` to a positive integer to keep only the newest N periodic checkpoints (best checkpoints are never deleted).

## Inference

```bash
uv run python generate.py --config generate.yaml
uv run python generate.py --config generate.yaml --ckpt outputs/checkpoints/best.pt
uv run python generate.py --config generate.yaml --run-name my_eval
```

Produces:

- `inference_outputs/<RUN_NAME>/raw_smiles.csv` — transpharmer stream output.
- `inference_outputs/<RUN_NAME>/scored_smiles.csv` — validated SMILES with `reward` column (deduplicated, sorted by reward).

## Custom rewards

Open `reward.py` and change **`compute_rewards`**. It must return a `list[float]` with one score per input SMILES (same length as `smiles_list`). The default combines **QED** and batch **Tanimoto diversity**.

## Config snippets

- **`RUN_NAME`** — prefix for checkpoint filenames and inference subfolder.
- **`OUTPUTS.DIR`** — root for training (`outputs`) or inference (`inference_outputs`) trees.
- **`DEVICE`** — e.g. `cuda:0` or `cpu`.
- **`RL_TRAIN.SAVE_EVERY`** — periodic checkpoint interval (set `0` to disable periodic saves except the final episode).
