#!/usr/bin/env bash
# scripts/install.sh
#
# One-shot installer for the poidh AI image detector Chrome extension.
# Downloads the model + ONNX-runtime WASM binaries from the v0.1.0 release
# and drops them into extension/{models,vendor}/. Then runs the local
# eval so you can verify everything works.
#
# Usage:
#     bash scripts/install.sh
#
# Tested on macOS (bash 3.2+/zsh) and Linux (bash 4+). Requires only:
#   - bash, curl, tar (already on every Mac)
#   - python3 + pip
#   - HuggingFace `datasets` library (auto-installs below)
#
# After install, load extension/ as an unpacked extension in Chrome.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_URL="https://github.com/JorgeQuijano/poidh-ai-detector/releases/download/v0.1.0/poidh-ai-detector-v0.1.0.tar.gz"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 1. Download the runtime asset bundle (model + WASM binaries, ~207MB compressed).
echo "==> Downloading poidh-ai-detector v0.1.0 (~207MB)"
curl -fL --progress-bar -o "$WORK/bundle.tar.gz" "$RELEASE_URL"

# 2. Extract into extension/.
echo "==> Extracting to extension/{models,vendor}/"
tar -xzf "$WORK/bundle.tar.gz" -C "$REPO_ROOT/extension"

# 3. Verify the model is the right one.
if [[ ! -f "$REPO_ROOT/extension/models/detector-int8-v0.1.0.onnx" ]]; then
    echo "!! model file missing after extract" >&2
    exit 1
fi
SIZE=$(du -k "$REPO_ROOT/extension/models/detector-int8-v0.1.0.onnx" | awk '{print $1}')
if (( SIZE < 290000 )); then
    echo "!! model file looks too small (${SIZE}KB); expected ~292MB" >&2
    exit 1
fi

# 4. Optional: run the local eval to verify everything works.
if [[ -d "$REPO_ROOT/eval/dataset/ai" ]] && [[ -d "$REPO_ROOT/eval/dataset/real" ]]; then
    echo "==> Eval dataset found at eval/dataset/. Running local eval..."
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c "import numpy, onnxruntime, PIL" >/dev/null 2>&1; then
            python3 "$REPO_ROOT/scripts/local-eval.py" || echo "(eval failed — extension may still work)"
        else
            echo "    (skipped: pip install -r requirements-eval.txt to run the eval)"
        fi
    else
        echo "    (skipped: python3 not found)"
    fi
else
    echo "==> No eval dataset at eval/dataset/{ai,real} — skipping local eval."
    echo "    (extension is still installed and ready to load in Chrome)"
fi

# 5. Print the next step.
cat <<EOF

===========================================================
✅ poidh-ai-detector v0.1.0 installed.

NEXT STEP: load the extension in Chrome
   1. Open chrome://extensions/
   2. Toggle "Developer mode" (top right)
   3. Click "Load unpacked"
   4. Select: $REPO_ROOT/extension

That's it. Browse any image-heavy site and you'll see overlay
badges on every detected image (AI 0.87 / REAL 0.92 / UNCERTAIN).
===========================================================
EOF
