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

## Training

```bash
uv run python train.py --config train.yaml
# Optional: override starting weights
uv run python train.py --config train.yaml --ckpt weights/guacamol_nc.pt
```

**CO2 objective:** `MGW_OBJECTIVE=co2 uv run python train.py --config train.yaml`

Reward, penalties, and RL drivers (KL, entropy, schedules): [`objectives/CO2_OBJECTIVE.md`](objectives/CO2_OBJECTIVE.md).

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
