# HPC: CO₂ binding RL (clone → generate proposals)

Package the RL generator for Isaac / any Slurm GPU node. Bayesian AL + DFT live in `CO2_IE_Dataset` (Phase 2).

## Clone

```bash
git clone git@github.com:dyashton/Molecule_Generation_Workflow.git
cd Molecule_Generation_Workflow
git checkout co2-binding-hpc
uv sync
```

Optional sibling for pack source / later DFT:

```bash
# next to MGW
git clone <your-co2-ie-dataset-remote> ../CO2_IE_Dataset
```

## Binding pack (~260 MB joblibs — not in git)

```bash
# From sibling monorepo (laptop / shared FS):
bash scripts/fetch_binding_pack.sh

# Or from a release tarball on scratch:
bash scripts/fetch_binding_pack.sh /path/to/binding_predictor.tar.gz
```

Creates `artifacts/binding_predictor/pack.joblib` (+ heads).

## Prior checkpoint + vocab

Shipped on this branch (when pushed):

- `weights/guacamol_nc_selfies_medium_co2.pt`
- `data/mixed_selfies/vocab.json`

If missing after clone, copy from a machine that has them, or override `--ckpt`. Binding pack joblibs are **not** in git (~70 MB tarball):

```bash
bash scripts/pack_binding_artifacts.sh   # on the machine with CO2_IE_Dataset
# upload artifacts/binding_predictor.tar.gz to scratch / GitHub Release
bash scripts/fetch_binding_pack.sh /path/to/binding_predictor.tar.gz
```

## Smoke test (CPU or GPU)

```bash
export MGW_OBJECTIVE=co2_binding_surrogate
export MGW_BINDING_PACK=$PWD/artifacts/binding_predictor/pack.joblib
export MGW_BINDING_HEAD=ie

uv run python scripts/export_proposals.py \
  --config generate_co2_binding_smoke.yaml \
  --n 64 \
  --out artifacts/discovery/smoke_proposals.csv
```

## Full generate → proposals.csv (AL/BO handoff)

```bash
bash scripts/run_co2_binding_generate.sh
# or Slurm:
sbatch scripts/hpc/sbatch_rl_generate.sh
```

Writes `artifacts/discovery/proposals.csv` with:

`SMILES,reward,reward_head,ood_gated,strong_prob,binding,uncertainty,ie,be,begm`

## Train (optional fine-tune)

```bash
bash scripts/run_co2_binding_rl.sh
# or:
sbatch scripts/hpc/sbatch_rl_train.sh
```

See [REWARD_CONTRACT.md](REWARD_CONTRACT.md).

## Scratch layout

```text
$SCRATCH/mgw_co2_binding/
  artifacts/binding_predictor/   # fetched pack
  outputs/                       # checkpoints
  artifacts/discovery/           # proposals.csv
  logs/
```

Point `SCRATCH_ROOT` / `MGW_SCRATCH` at that directory in the sbatch scripts.
