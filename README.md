# poidh-ai-detector

A Chrome extension (MV3) that detects AI-generated images **entirely on-device** — no image data is ever sent to a remote server.

Built for the [poidh bounty #323](https://poidh.xyz/arbitrum/bounty/323): *"local AI challenge: AI image detector for Chrome"*. Hard target: **≥ 75.0% balanced accuracy** at a **65% confidence threshold** on a private benchmark, with all inference inside the extension sandbox.

## How it works (architecture)

```
+--------------------+        +-----------------------+        +-------------------+
|  content script    |  img   |  background service  |  msg   |  OFFSCREEN doc    |
|  - MutationObserver|--bits->|  worker (router)     |------->|  - onnxruntime-web|
|  - Plan 3 heuristic| (PNG/  |  - heuristic pre-flt |        |  - WASM backend   |
|  - overlay badge   |  JPEG) |  - dispatch -> offscr|        |  - 224x224 RGB    |
+--------------------+        +-----------------------+        +-------------------+
                                                                    |
                                                                    v
                                                          balanced accuracy score
                                                            {ai, real, uncertain}
```

- **WASM-only** — no WebGPU dependency. Runs on stock Chromium / worst-case eval environments.
- **Plan 1 + Plan 3 hybrid pipeline**:
  - **Tier 1 — heuristic pre-filter** (C2PA manifest, EXIF `Software` tag, PNG text chunks, known generator string match) — fast verdict on metadata-rich images. Never declares "real" on heuristic alone (only positive AI evidence).
  - **Tier 2 — neural inference** via `onnxruntime-web` WASM in the offscreen document. 224x224 RGB input, ImageNet normalization, int8 quantized.
  - Deterministic verdict at the 65% confidence threshold.
- **On-device-only** — image bytes never leave the user's browser. No telemetry, no remote calls. The linter enforces no remote URLs in shipped code.

## Layout

```
poidh-ai-detector/
├── extension/                     # the Chrome extension (load this as unpacked)
│   ├── manifest.json
│   ├── background/service-worker.js     # MV3 router
│   ├── offscreen/                       # hosts the WASM session (workers can't)
│   ├── content/                         # MutationObserver + overlay badge
│   ├── heuristics/heuristic-filter.js   # Plan 3 pre-filter
│   ├── vendor/ort.min.js                # onnxruntime-web JS shim (committed)
│   ├── vendor/ort-wasm-simd-threaded.*  # WASM binaries (gitignored, fetched)
│   ├── models/                          # ONNX int8 model (gitignored, fetch)
│   └── icons/
├── eval/                                # local regression test (mirrors the
│   ├── index.html, runner.js, ...       #   maintainer's expected evaluation)
│   └── dataset/                         # gitignored public evaluation images
├── scripts/                             # fetch-model.sh, check-size.sh, lint
├── .github/workflows/ci.yml
├── LICENSE
└── README.md
```

## Quick start (development)

```bash
git clone https://github.com/<owner>/poidh-ai-detector && cd poidh-ai-detector

# 1. Fetch WASM binaries and (optionally) a placeholder model.
bash scripts/fetch-model.sh

# 2. Drop a real 224x224 ONNX int8 model at extension/models/detector-int8-v0.1.0.onnx
#    (see "Choosing a model" below)

# 3. Load the extension unpacked in Chrome:
#    chrome://extensions -> Developer mode -> "Load unpacked" -> select extension/
# 4. Browse — small overlay appears on every detected image.
# 5. Run the eval harness:
bash scripts/eval-server.sh
#    open http://127.0.0.1:8090/eval/ in Chrome
```

## Choosing a model

The neural tier is a thin wrapper around an ONNX int8 model. We evaluated candidate
detectors — the current shortlist:

| Model | Why consider | Approx ONNX-int8 size |
|---|---|---|
| **UnivFD** (CLIP-L/14 + linear probe) | Top-tier cross-generator gen detection | ~30–60MB |
| **DRCT** | Diffusion-specific, small head | ~10–30MB |
| **DeteCT** | Strong cross-model detection | ~20–50MB |
| **Gram-Net / FreqNet** | Cheap, frequency-domain, very small | ~2–10MB |

Drop one of these (exported to ONNX, int8 quantized, input 224×224×3 RGB, single
output sigmoid or 2-logit softmax) at `extension/models/detector-int8-v0.1.0.onnx`.
The pipeline is tolerant of either output shape.

## Eval bar

The bundled `eval/` page runs the same pipeline the extension ships with over a
folder of images and reports balanced accuracy at the 65% confidence threshold.
It's the local regression test *and* the same flow we'd expect the maintainer to
use. Folder layout: `eval/dataset/real/` and `eval/dataset/ai/`.

The offline eval harness is the local proof; the maintainer will run their own
private benchmark on the same model.

## Constraints

- **WASM-only** — no WebGPU, no SharedArrayBuffer, no COOP/COEP. Runs on stock
  Chromium in a Docker-style eval environment.
- **300MB ceiling** — `scripts/check-size.sh` enforces the total extension size
  in CI. Current vendor footprint at ~40MB; we have ~260MB of headroom for the
  neural model.
- **No remote calls** — `scripts/lint.js` scans shipped code for any HTTP URL
  and rejects it (privacy guarantee).

## License

MIT — see [LICENSE](./LICENSE).
