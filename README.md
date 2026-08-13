# poidh-ai-detector

A Chrome extension (MV3) that detects AI-generated images **entirely on-device** — no image data is ever sent to a remote server.

Built for the [poidh bounty #323](https://poidh.xyz/arbitrum/bounty/323): *"local AI challenge: AI image detector for Chrome"*. Hard target: **≥ 75.0% balanced accuracy** at a **65% confidence threshold** on a private benchmark, with all inference inside the extension sandbox.

## Status (v0.1.0)

| Metric | Value |
|---|---|
| Model | UnivFD CLIP ViT-L/14 + linear probe, int8 quantized (~292MB) |
| Backbone | 427.6M params (CLIP ViT-L/14, frozen) |
| Probe | 768→2 linear layer (real/fake), exported as 2-class ONNX |
| Total extension size | ~330MB (under 340MB CI ceiling) |
| Inference backend | onnxruntime-web 1.27.0, WASM-only, no WebGPU |
| Local eval result | **94.00% balanced accuracy** on frp94/progan_val (50/50 split) — **PASS** |
| Honest caveat | The local eval is on a dataset *similar to* UnivFD's training distribution. The maintainer's private eval may be on different generators; cross-generator performance is unverified. |

## One-command install

```bash
git clone https://github.com/JorgeQuijano/poidh-ai-detector.git
cd poidh-ai-detector
bash scripts/install.sh
```

Then load `extension/` as an unpacked extension in Chrome:

```
chrome://extensions/  →  Developer mode (top right)  →  Load unpacked  →  select extension/
```

That's it. The installer downloads the 207MB runtime bundle (model + WASM binaries) from the GitHub release — **no PyTorch, no 3GB dependencies, no build step**.

## How it works

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
  - **Tier 2 — neural inference** via `onnxruntime-web` WASM in the offscreen document. **CLIP ViT-L/14 input contract**: shortest-side resize to 256, center-crop 224, normalize with CLIP stats (`mean=[0.4815, 0.4578, 0.4082]`, `std=[0.2686, 0.2613, 0.2758]`).
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
│   ├── vendor/ort-wasm-simd-threaded.*  # WASM binaries (downloaded by install.sh)
│   ├── models/                          # ONNX int8 model (downloaded by install.sh)
│   │   └── detector-int8-v0.1.0.onnx    # 292MB; SHA-256 3d986aed...f37d
│   └── icons/
├── eval/                                # local regression test
│   ├── index.html, runner.js            # the maintainer-style eval UI
│   ├── preprocess.js                    # CLIP-correct preprocessing
│   ├── onnxruntime-config.js            # pins WASM execution providers
│   └── dataset/                         # gitignored eval images (real + AI)
├── scripts/                             # install / fetch-model / build-model / eval
│   ├── install.sh                       # ONE-SHOT installer (downloads release assets)
│   ├── fetch-model.sh                   # WASM-only fetcher (legacy)
│   ├── build-model.py                   # PyTorch build (maintainers only)
│   ├── local-eval.py                    # Python eval runner (no browser needed)
│   └── lint.js, check-size.sh
├── .github/workflows/ci.yml
├── LICENSE
└── README.md
```

## Eval bar

The bundled `eval/index.html` runs the same pipeline the extension ships with over a folder of images and reports balanced accuracy at the 65% confidence threshold.

Our local measurement:
- **Dataset:** `frp94/progan_val` (ProGAN validation, matches UnivFD's training distribution)
- **Size:** 50 real + 50 AI
- **Preprocessing:** shortest-side resize 256, center-crop 224, CLIP normalization
- **Result:** 94.00% balanced accuracy (47/50 AI detected, 47/50 real detected) — **PASS**

The maintainer will run their own private benchmark; the local eval is a *necessary* (not sufficient) signal. Cross-generator performance is the unknown.

To reproduce the eval locally:

```bash
pip install -r requirements-eval.txt
python3 scripts/local-eval.py
# → 94.00% balanced accuracy, PASS
```

To populate `eval/dataset/`, see [`scripts/install.sh`](./scripts/install.sh) — it auto-fetches a balanced ProGAN val sample if you have the `datasets` Python package installed.

## Constraints

- **WASM-only** — no WebGPU, no SharedArrayBuffer, no COOP/COEP. Runs on stock Chromium in a Docker-style eval environment.
- **340MB ceiling** — `scripts/check-size.sh` enforces the total extension size in CI. UnivFD int8 + onnxruntime-web WASM = ~330MB. Override with `MAX_MB=...` if you need stricter.
- **No remote calls** — `scripts/lint.js` scans shipped code for any HTTP URL and rejects it (privacy guarantee).

## For maintainers (re-building the model)

The shipped `detector-int8-v0.1.0.onnx` is built from UniversalFakeDetect (CLIP:ViT-L/14 + linear probe). To rebuild it from sources:

```bash
# Big install (~3GB): PyTorch CPU + ONNX + OpenAI CLIP
pip install -r requirements-build.txt

# Build the model (~10 minutes)
python3 scripts/build-model.py
```

This is **not required** for users — the GitHub release already ships the pre-built model. The dependencies above are only for maintainers who want to rebuild or fine-tune.

## License

MIT — see [LICENSE](./LICENSE).
