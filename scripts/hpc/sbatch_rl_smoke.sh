#!/bin/bash
#SBATCH --job-name=mgw_co2_smoke
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_rl_smoke_%j.out
#SBATCH --error=logs/slurm_rl_smoke_%j.err

set -euo pipefail
# See sbatch_rl_generate.sh — Slurm spool $0 is not the repo path.
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

OUT="${MGW_SMOKE_OUT:-$ROOT/artifacts/discovery/smoke_proposals.csv}"
echo "[sbatch smoke] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-none} -> $OUT"
uv run python scripts/export_proposals.py \
  --config generate_co2_binding_smoke.yaml \
  --n "${MGW_SMOKE_N:-64}" \
  --out "$OUT"
echo "[sbatch smoke] done rows=$(wc -l < "$OUT")"
