#!/bin/bash
#SBATCH --job-name=mgw_co2_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_rl_gen_%j.out
#SBATCH --error=logs/slurm_rl_gen_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs artifacts/discovery

if [[ ! -f artifacts/binding_predictor/pack.joblib ]]; then
  if [[ -n "${MGW_BINDING_TARBALL:-}" ]]; then
    bash scripts/fetch_binding_pack.sh "$MGW_BINDING_TARBALL"
  else
    bash scripts/fetch_binding_pack.sh
  fi
fi

export MGW_OBJECTIVE=co2_binding_surrogate
export MGW_BINDING_PACK="$ROOT/artifacts/binding_predictor/pack.joblib"
export MGW_BINDING_HEAD=ie
export MGW_BINDING_STRONG_BONUS="${MGW_BINDING_STRONG_BONUS:-0.5}"
export MGW_PROPOSALS_OUT="${MGW_PROPOSALS_OUT:-$ROOT/artifacts/discovery/proposals.csv}"

echo "[sbatch] host=$(hostname) -> $MGW_PROPOSALS_OUT"
bash scripts/run_co2_binding_generate.sh
