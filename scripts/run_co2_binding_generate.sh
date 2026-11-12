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

OUT="${MGW_PROPOSALS_OUT:-$ROOT/artifacts/discovery/proposals.csv}"
uv run python scripts/export_proposals.py \
  --config generate_co2_binding.yaml \
  --out "$OUT" \
  "$@"
echo "[generate] proposals -> $OUT"
