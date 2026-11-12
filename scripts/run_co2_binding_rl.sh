#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f artifacts/binding_predictor/pack.joblib ]]; then
  bash scripts/fetch_binding_pack.sh
fi

export MGW_OBJECTIVE=co2_binding_surrogate
export MGW_BINDING_PACK="${MGW_BINDING_PACK:-$ROOT/artifacts/binding_predictor/pack.joblib}"
export MGW_BINDING_HEAD="${MGW_BINDING_HEAD:-ie}"
export MGW_BINDING_STRONG_BONUS="${MGW_BINDING_STRONG_BONUS:-0.5}"

uv run python train.py --config train_selfies_co2_binding.yaml "$@"
