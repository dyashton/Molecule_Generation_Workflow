#!/bin/bash
#SBATCH --job-name=mgw_co2_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_rl_train_%j.out
#SBATCH --error=logs/slurm_rl_train_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs

SCRATCH_ROOT="${MGW_SCRATCH:-${SCRATCH:-$ROOT}/mgw_co2_binding}"
mkdir -p "$SCRATCH_ROOT/artifacts/binding_predictor" "$SCRATCH_ROOT/outputs" "$SCRATCH_ROOT/logs"

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

echo "[sbatch] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
bash scripts/run_co2_binding_rl.sh
