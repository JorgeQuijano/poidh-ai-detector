# poidh-ai-detector

A Chrome extension (MV3) that detects AI-generated images **entirely on-device** — no image data is ever sent to a remote server.

Built for the [poidh bounty #323](https://poidh.xyz/arbitrum/bounty/323): *"local AI challenge: AI image detector for Chrome"*. Hard target: **≥ 75.0% balanced accuracy** at a **65% confidence threshold** on a private benchmark, with all inference inside the extension sandbox.

## How it works (architecture)

```
+--------------------+        +-----------------------+        +-------------------+
|  content script    |  img   |  background service  |  msg   |  OFFSCREEN doc    |
|  - MutationObserver|--bits->|  worker (router)     |------->|  - onnxruntime-web|
|  - plan3 heuristic | (PNG/  |  - heuristic pre-flt |        |  - WASM backend   |
|  - overlay badge   |  JPEG) |  - dispatch -> offscr|        |  - 224x224 RGB    |
+--------------------+        +-----------------------+        +-------------------+
                                                                    |
                                                                    v
                                                          balanced accuracy score
                                                            {ai, real, uncertain}
```

- **WASM-only** — no WebGPU dependency. Runs on stock Chromium / worst-case eval environments.
- **Three-tier pipeline** (per the chosen plan):
  - Tier 1: Plan-3 heuristic pre-filter (C2PA manifest, EXIF, JPEG color stats, entropy) — fast verdict on metadata-rich images.
  - Tier 2: ONNX image classifier (UnivFD / DRCT / FreqNet, int8 quantized) — neural fallback for anything uncertain.
  - Tier 3 (future): bundled fingerprint of known generator outputs.
- **On-device-only** — image bytes never leave the user's browser. No telemetry, no remote calls.

## Layout

```
poidh-ai-detector/
├── extension/                   # the Chrome extension (load this as unpacked)
│   ├── manifest.json
│   ├── background/
│   │   └── service-worker.js
│   ├── offscreen/
│   │   └── offscreen.html + .js   # hosts the onnxruntime-web WASM session
│   ├── content/
│   │   ├── content-script.js      # observes <img> inserts, renders overlay
│   │   └── overlay.css
│   ├── heuristics/
│   │   └── heuristic-filter.js    # Plan 3 pre-filter
│   ├── vendor/
│   │   └── ort.min.js             # onnxruntime-web, pinned version
│   └── icons/
├── eval/                         # eval harness (mirrors the maintainer's expected test)
│   ├── index.html
│   ├── runner.js
│   └── dataset/                  # gitignored public evaluation images
├── models/                       # gitignored ONNX models (downloaded by scripts/fetch-model.sh)
├── scripts/
│   ├── fetch-model.sh
│   └── check-size.sh
├── .github/workflows/ci.yml
├── LICENSE
└── README.md
```

## Quick start (development)

1. `git clone https://github.com/<owner>/poidh-ai-detector && cd poidh-ai-detector`
2. Download a model: `bash scripts/fetch-model.sh` (puts an ONNX int8 in `models/`)
3. Load the extension unpacked in Chrome:
   `chrome://extensions` → enable "Developer mode" → "Load unpacked" → select `extension/`
4. Browse — small overlay appears on every detected image.
5. Run the eval harness: `cd eval && python3 -m http.server 8080` → open `http://localhost:8080/`

## Eval bar

The bundled `eval/` page runs the same pipeline the extension ships with over a folder of images and reports balanced accuracy at the 65% confidence threshold. It's the local regression test *and* the same flow we'd expect the maintainer to use.

## License

MIT — see [LICENSE](./LICENSE).
