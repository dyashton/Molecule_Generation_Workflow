#!/usr/bin/env bash
# Install binding_predictor joblibs into artifacts/binding_predictor/.
# Usage:
#   bash scripts/fetch_binding_pack.sh
#   bash scripts/fetch_binding_pack.sh /path/to/binding_predictor.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MGW_BINDING_DIR:-$ROOT/artifacts/binding_predictor}"
SRC_DEFAULT="$ROOT/../CO2_IE_Dataset/artifacts/binding_predictor"
NEEDED=(pack.joblib ie.joblib be.joblib penalty.joblib strong.joblib)

mkdir -p "$DEST"

copy_dir() {
  local src="$1"
  echo "[fetch] copying from $src -> $DEST"
  for f in "${NEEDED[@]}"; do
    if [[ ! -f "$src/$f" ]]; then
      echo "[fetch] missing required $src/$f" >&2
      exit 1
    fi
    cp -f "$src/$f" "$DEST/$f"
  done
  # Optional extras
  for f in metrics.json oof.csv ablations.json begm.joblib begm_direct.joblib; do
    [[ -f "$src/$f" ]] && cp -f "$src/$f" "$DEST/$f" || true
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,8p' "$0"
  exit 0
fi

if [[ $# -ge 1 ]]; then
  ARCHIVE="$1"
  if [[ -d "$ARCHIVE" ]]; then
    copy_dir "$ARCHIVE"
  elif [[ -f "$ARCHIVE" ]]; then
    tmp="$(mktemp -d)"
    echo "[fetch] extracting $ARCHIVE"
    tar -xzf "$ARCHIVE" -C "$tmp"
    # Accept either flat or nested binding_predictor/
    if [[ -f "$tmp/pack.joblib" ]]; then
      copy_dir "$tmp"
    elif [[ -f "$tmp/binding_predictor/pack.joblib" ]]; then
      copy_dir "$tmp/binding_predictor"
    else
      found="$(find "$tmp" -name pack.joblib | head -1 || true)"
      if [[ -z "$found" ]]; then
        echo "[fetch] pack.joblib not found in archive" >&2
        exit 1
      fi
      copy_dir "$(dirname "$found")"
    fi
    rm -rf "$tmp"
  else
    echo "[fetch] not a file or directory: $ARCHIVE" >&2
    exit 1
  fi
elif [[ -d "$SRC_DEFAULT" ]]; then
  copy_dir "$SRC_DEFAULT"
else
  echo "[fetch] no sibling pack at $SRC_DEFAULT and no archive given." >&2
  echo "Build a tarball with: bash scripts/pack_binding_artifacts.sh" >&2
  exit 1
fi

# Always rewrite from MGW root (callers may invoke this script from a sibling cwd).
cd "$ROOT"
uv run python scripts/rewrite_binding_pack_paths.py --dest "$DEST"
ls -lh "$DEST"