#!/usr/bin/env python3
"""
scripts/local-eval.py

Local regression test for the extension's neural detector. Scans
eval/dataset/{real,ai}/*.jpg, runs them through the ONNX int8 model with
the same CLIP preprocessing pipeline as the extension, and reports balanced
accuracy at the 65% confidence threshold.

This is the maintainer-style eval, runnable from Python directly (no
browser needed). For the in-browser harness, see eval/index.html.

Run from the repo root:
    python3 scripts/local-eval.py
"""

import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

# CLIP ViT-L/14 input contract (matches UniversalFakeDetect validate.py):
#   1. shortest-side resize to 256
#   2. center-crop 224x224
#   3. CLIP normalization (NOT ImageNet)
INPUT_SIZE = 224
RESIZE_SIZE = 256
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
THRESHOLD = 0.65


def preprocess(pil_img):
    """Rescale to 256 shortest-side, center-crop 224, CLIP-normalize."""
    w, h = pil_img.size
    scale = RESIZE_SIZE / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    pil_img = pil_img.resize((nw, nh), Image.BICUBIC)
    off_x = (nw - INPUT_SIZE) // 2
    off_y = (nh - INPUT_SIZE) // 2
    pil_img = pil_img.crop((off_x, off_y, off_x + INPUT_SIZE, off_y + INPUT_SIZE))
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    # HWC -> NCHW
    return arr.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "extension/models/detector-int8-v0.1.0.onnx")
    dataset = os.path.join(repo_root, "eval/dataset")

    if not os.path.isfile(model_path):
        raise SystemExit(f"model not found at {model_path} — run scripts/fetch-model.sh")

    print("Loading model...", flush=True)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("  input:", sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
    print("  output:", sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

    results = []
    t0 = time.time()
    for cls in ["ai", "real"]:
        d = os.path.join(dataset, cls)
        if not os.path.isdir(d):
            raise SystemExit(f"missing class folder: {d}")
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(d, fn)
            try:
                img = Image.open(path)
                t = preprocess(img)
                out = sess.run(None, {"image": t})[0]
                if out.shape == (1, 1):
                    # 1-class sigmoid output.
                    p_ai = float(1.0 / (1.0 + np.exp(-out[0, 0])))
                else:
                    # 2-class softmax output.  Index 1 = "fake".
                    logit_real, logit_ai = out[0]
                    m = max(logit_real, logit_ai)
                    p_ai = float(np.exp(logit_ai - m) / (np.exp(logit_real - m) + np.exp(logit_ai - m)))
                results.append({
                    "file": fn,
                    "truth": cls,
                    "p_ai": round(p_ai, 3),
                    "pred": "ai" if p_ai >= THRESHOLD else "real",
                })
            except Exception as e:
                results.append({"file": fn, "truth": cls, "error": str(e)})
    elapsed = time.time() - t0

    # Confusion matrix.
    cm = {"ai": {"tp": 0, "fn": 0, "fp": 0, "tn": 0}, "real": {"tp": 0, "fn": 0, "fp": 0, "tn": 0}}
    for r in results:
        if "error" in r:
            continue
        if r["truth"] == "ai":
            if r["pred"] == "ai":
                cm["ai"]["tp"] += 1
            else:
                cm["ai"]["fn"] += 1
            if r["pred"] == "real":
                cm["real"]["fp"] += 1
            else:
                cm["real"]["tn"] += 1
        else:
            if r["pred"] == "real":
                cm["real"]["tp"] += 1
            else:
                cm["real"]["fn"] += 1
            if r["pred"] == "ai":
                cm["ai"]["fp"] += 1
            else:
                cm["ai"]["tn"] += 1

    tpr_ai = cm["ai"]["tp"] / max(1, cm["ai"]["tp"] + cm["ai"]["fn"])
    tpr_real = cm["real"]["tp"] / max(1, cm["real"]["tp"] + cm["real"]["fn"])
    balanced = (tpr_ai + tpr_real) / 2

    summary = {
        "n": len(results),
        "elapsed_s": round(elapsed, 1),
        "threshold": THRESHOLD,
        "confusion_matrix": cm,
        "tpr_ai": round(tpr_ai, 3),
        "tpr_real": round(tpr_real, 3),
        "balanced_accuracy": f"{balanced * 100:.2f}%",
        "verdict": "PASS" if balanced >= 0.75 else "FAIL",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
