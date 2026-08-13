#!/usr/bin/env python3
"""
scripts/build-model.py

Exports UniversalFakeDetect (CLIP:ViT-L/14 + linear probe) to a 2-class
ONNX int8 model suitable for the extension.

Output: extension/models/detector-int8-v0.1.0.onnx

Steps:
  1. Load CLIP ViT-L/14 from `clip` package (downloads on first run).
  2. Load the linear probe from UniversalFakeDetect's pretrained_weights.
  3. Compose a single 2-class CLIPModel (768 -> 2 linear).
  4. Export to ONNX fp32 with embedded weights.
  5. Simplify the graph with onnxsim.
  6. Quantize weights to int8 (dynamic quantization).

Run from the repo root:
    python3 scripts/build-model.py
"""

import os
import sys
import time

# 1. Load model
import torch
import torch.nn as nn
import clip   # openai-clip

CHANNELS = {"RN50": 1024, "ViT-L/14": 768}


class CLIPModel(nn.Module):
    """CLIP backbone + linear probe head, with a 2-class output layout."""

    def __init__(self, name: str, num_classes: int = 2):
        super().__init__()
        # The clip package downloads the weights to ~/.cache/clip on first call.
        # We pass download_root so it's reproducible.
        self.model, _ = clip.load(
            name, device="cpu", download_root="/tmp/clip_models"
        )
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x):
        return self.fc(self.model.encode_image(x))


def main():
    name = "ViT-L/14"
    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "extension",
        "models",
        "detector-int8-v0.1.0.onnx",
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("==> Building CLIPModel (CLIP:ViT-L/14 + 768->2 linear probe)", flush=True)
    m = CLIPModel(name, num_classes=2)

    # Load the original 1-class probe and adapt it to 2-class.
    # Original probe: single logit.  sigmoid(logit) > 0.5  =>  label 1 (fake).
    # To expose a 2-class output [real, fake] where index = label:
    #   weight[0] = -fc_w  (real class)
    #   weight[1] = +fc_w  (fake class)
    #   bias[0]   = -fc_b
    #   bias[1]   = +fc_b
    # argmax of the 2 logits gives the predicted label.
    fc_path = "/tmp/UniversalFakeDetect/pretrained_weights/fc_weights.pth"
    if not os.path.isfile(fc_path):
        print(
            f"!! Missing {fc_path}.",
            "Clone https://github.com/Yuheng-Li/UniversalFakeDetect to /tmp first.",
            file=sys.stderr,
        )
        sys.exit(1)
    fc_w = torch.load(fc_path, map_location="cpu", weights_only=False)
    m.fc.weight.data[0] = -fc_w["weight"].squeeze(0)
    m.fc.weight.data[1] = fc_w["weight"].squeeze(0)
    m.fc.bias.data[0] = -fc_w["bias"].item()
    m.fc.bias.data[1] = fc_w["bias"].item()
    m.eval()

    # 2. Export to ONNX fp32 with embedded weights.
    print("==> Exporting ONNX fp32 (with embedded weights)...", flush=True)
    tmp_fp32 = out_path + ".fp32.tmp"
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            m,
            x,
            tmp_fp32,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    import onnx

    model = onnx.load(tmp_fp32, load_external_data=True)
    onnx.save(model, tmp_fp32, save_as_external_data=False)
    print(
        f"   fp32 size: {os.path.getsize(tmp_fp32) / 1e6:.1f} MB", flush=True
    )

    # 3. Simplify.
    print("==> Simplifying graph with onnxsim...", flush=True)
    import onnxsim

    simplified, ok = onnxsim.simplify(tmp_fp32, perform_optimization=False)
    if not ok:
        print("!! onnxsim failed; falling back to the unsimplified model", flush=True)
        simplified = onnx.load(tmp_fp32)
    onnx.save(simplified, tmp_fp32, save_as_external_data=False)

    # 4. Quantize weights to int8.
    print("==> Quantizing weights to int8 (dynamic quantization)...", flush=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType

    tmp_int8 = out_path + ".int8.tmp"
    t0 = time.time()
    quantize_dynamic(
        tmp_fp32,
        tmp_int8,
        weight_type=QuantType.QInt8,
    )
    print(f"   quantized in {time.time() - t0:.1f}s", flush=True)
    print(
        f"   int8 size: {os.path.getsize(tmp_int8) / 1e6:.1f} MB", flush=True
    )

    # 5. Move to final path.
    os.replace(tmp_int8, out_path)
    os.remove(tmp_fp32)
    print(f"==> Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
