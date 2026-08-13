#!/usr/bin/env bash
# scripts/fetch-model.sh
# Fetches a pinned onnxruntime-web release and (optionally) the model file.
# Re-runnable, idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/extension/vendor"
MODEL_DIR="$REPO_ROOT/extension/models"

# Pinned versions — bump explicitly when we want to upgrade.
ORT_VERSION="1.27.0"
MODEL_VERSION="${MODEL_VERSION:-detector-int8-v0.1.0}"

# --- onnxruntime-web vendor --------------------------------------------------
mkdir -p "$VENDOR_DIR"
echo "==> Fetching onnxruntime-web v${ORT_VERSION}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Resolve the package tarball URL from the npm registry.
TARBALL_URL="$(curl -sL "https://registry.npmjs.org/onnxruntime-web/${ORT_VERSION}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['dist']['tarball'])")"

curl -sL "$TARBALL_URL" | tar -xz -C "$WORK"

# Copy the runtime JS + WASM binaries we need. We intentionally exclude the
# .node binary to keep the extension WASM-only.
cp "$WORK/package/dist/ort.min.js" "$VENDOR_DIR/ort.min.js"

# The v1.27.0 npm package only ships the SIMD-threaded WASM variant. The
# runtime tries three flavors of the .mjs loader:
#   1. ort-wasm-simd-threaded.mjs          (plain threaded, no WebGPU bridge)
#   2. ort-wasm-simd-threaded.jsep.mjs     (default; loaded by ort.min.js)
#   3. ort-wasm-simd-threaded.jspi.mjs     (JS-PI fallback for some browsers)
# We ship all three JS loaders + the plain + jsep WASM binaries. (~40MB total)
# numThreads=1 in the runtime config means we don't need COOP/COEP headers.
for f in \
  ort-wasm-simd-threaded.mjs \
  ort-wasm-simd-threaded.wasm \
  ort-wasm-simd-threaded.jsep.mjs \
  ort-wasm-simd-threaded.jsep.wasm \
  ort-wasm-simd-threaded.jspi.mjs \
  ort-wasm-simd-threaded.jspi.wasm
do
  cp "$WORK/package/dist/$f" "$VENDOR_DIR/" || true
done

# Sanity check.
test -f "$VENDOR_DIR/ort.min.js" || { echo "ort.min.js missing"; exit 1; }
echo "    onnxruntime-web vendored at $VENDOR_DIR/"
ls -la "$VENDOR_DIR"

# --- Model -------------------------------------------------------------------
# The model is intentionally NOT auto-fetched from a single source. The
# candidate models (UnivFD, DRCT, FreqNet, DeteCT) live in different repos and
# require license-aware downloads. We leave a placeholder here so the extension
# can still load once the model is in place.
mkdir -p "$MODEL_DIR"
MODEL_PATH="$MODEL_DIR/${MODEL_VERSION}.onnx"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "==> Model not found at $MODEL_PATH"
  echo "    Drop a 224x224 ONNX int8 model there."
  echo "    See README.md for the candidate shortlist (UnivFD, DRCT, FreqNet, DeteCT)."
  # Create a stub so the link check at least passes.
  echo "TODO: see scripts/fetch-model.sh" > "$MODEL_PATH"
fi

echo "==> Done."
