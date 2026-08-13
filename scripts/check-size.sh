#!/usr/bin/env bash
# scripts/check-size.sh
# Verifies the extension package (vendor + model + code) stays under the
# 300MB ceiling Jorge set as the "worst we can go" budget for the bounty.
# Runs in CI to fail PRs that bloat.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAX_MB="${MAX_MB:-300}"

usage_pretty() {
  local kb=$1
  awk -v kb="$kb" 'BEGIN { printf "%.1f MB", kb/1024.0 }'
}

echo "Extension package size check (max ${MAX_MB} MB)"
echo "------------------------------------------------"

total_kb=0
for d in "$REPO_ROOT/extension/vendor" "$REPO_ROOT/extension/models"; do
  if [[ -d "$d" ]]; then
    kb=$(du -sk "$d" | awk '{print $1}')
    printf "  %-40s %s\n" "${d#$REPO_ROOT/}" "$(usage_pretty "$kb")"
    total_kb=$((total_kb + kb))
  fi
done

# Code itself is negligible but check it.
code_kb=$(find "$REPO_ROOT/extension" -type f \
  ! -path "*/vendor/*" ! -path "*/models/*" -exec du -sk {} + | awk '{sum+=$1} END {print sum+0}')
total_kb=$((total_kb + code_kb))
printf "  %-40s %s\n" "extension (code only)" "$(usage_pretty "$code_kb")"

echo "------------------------------------------------"
printf "  TOTAL                                   %s\n" "$(usage_pretty "$total_kb")"

limit_kb=$((MAX_MB * 1024))
if (( total_kb > limit_kb )); then
  echo "FAIL: total ${total_kb}KB exceeds ${MAX_MB}MB budget."
  exit 1
fi
echo "OK: within budget."
