#!/usr/bin/env bash
# Build a portable tarball of binding_predictor joblibs for HPC fetch.
# Usage:
#   bash scripts/pack_binding_artifacts.sh
#   bash scripts/pack_binding_artifacts.sh /path/to/out.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${MGW_BINDING_SRC:-$ROOT/../CO2_IE_Dataset/artifacts/binding_predictor}"
OUT="${1:-$ROOT/artifacts/binding_predictor.tar.gz}"

if [[ ! -d "$SRC" ]]; then
  echo "missing source dir: $SRC" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
tmp="$(mktemp -d)"
mkdir -p "$tmp/binding_predictor"
for f in pack.joblib ie.joblib be.joblib penalty.joblib strong.joblib \
         metrics.json oof.csv ablations.json begm.joblib begm_direct.joblib; do
  [[ -f "$SRC/$f" ]] && cp -f "$SRC/$f" "$tmp/binding_predictor/$f" || true
done
tar -czf "$OUT" -C "$tmp" binding_predictor
rm -rf "$tmp"
ls -lh "$OUT"
echo "Upload this tarball to scratch / GitHub Releases, then:"
echo "  bash scripts/fetch_binding_pack.sh $OUT"