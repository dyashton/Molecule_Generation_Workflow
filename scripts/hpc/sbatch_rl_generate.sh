#!/bin/bash
#SBATCH --job-name=mgw_co2_gen
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=14G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_rl_gen_%j.out
#SBATCH --error=logs/slurm_rl_gen_%j.err

set -euo pipefail
# Slurm often copies the batch script to a spool path ($0 ≠ repo). Prefer:
#   1) MGW_ROOT export  2) --chdir / PWD  3) script-relative  4) SUBMIT_DIR
if [[ -n "${MGW_ROOT:-}" && -f "${MGW_ROOT}/scripts/run_co2_binding_generate.sh" ]]; then
  ROOT="$MGW_ROOT"
elif [[ -f "${PWD}/scripts/run_co2_binding_generate.sh" ]]; then
  ROOT="$PWD"
elif [[ -f "$(cd "$(dirname "$0")/../.." && pwd)/scripts/run_co2_binding_generate.sh" ]]; then
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_co2_binding_generate.sh" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  echo "[sbatch] cannot locate MGW root (PWD=$PWD \$0=$0)" >&2
  exit 1
fi
cd "$ROOT"
mkdir -p logs artifacts/discovery

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCHDIR:-/lustre/isaac24/scratch/$USER}/DFT_Calculations/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

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

echo "[sbatch] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-none} -> $MGW_PROPOSALS_OUT"
bash scripts/run_co2_binding_generate.sh
