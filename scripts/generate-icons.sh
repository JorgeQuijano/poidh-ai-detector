#!/usr/bin/env bash
# scripts/generate-icons.sh
# Generates placeholder icon-16.png, icon-48.png, icon-128.png for the
# extension. Falls back to a minimal solid-color PNG if ImageMagick is missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ICON_DIR="$REPO_ROOT/extension/icons"
mkdir -p "$ICON_DIR"

if command -v convert >/dev/null 2>&1; then
  for sz in 16 48 128; do
    convert -size ${sz}x${sz} \
      -define gradient:angle=135 \
      gradient:'#2563eb-#0ea5e9' \
      -gravity center -font DejaVu-Sans-Bold -pointsize $((sz/2)) \
      -fill white -annotate +0+0 'P' \
      "$ICON_DIR/icon-${sz}.png"
  done
  echo "Wrote icons via ImageMagick."
else
  python3 - "$ICON_DIR" <<'PY'
import struct, zlib, sys, pathlib
ICON_DIR = pathlib.Path(sys.argv[1])
ICON_DIR.mkdir(parents=True, exist_ok=True)

def make_png(w, h, rgb=(37, 99, 235)):
    raw = b""
    for y in range(h):
        raw += b"\x00"
        for x in range(w):
            raw += bytes(rgb)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

for sz in (16, 48, 128):
    (ICON_DIR / f"icon-{sz}.png").write_bytes(make_png(sz, sz))
print("Wrote placeholder icons at", ICON_DIR)
PY
fi
ls -la "$ICON_DIR"
