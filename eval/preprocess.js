// Preprocessing matching UniversalFakeDetect's validate.py for CLIP:ViT-L/14:
//   1. resize shortest side to 256
//   2. center crop 224x224
//   3. normalize with CLIP stats (not ImageNet)

const INPUT_SIZE = 224;
const RESIZE_SIZE = 256;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];

function bitmapToTensor(bitmap) {
  // Step 1: resize shortest side to RESIZE_SIZE
  const w0 = bitmap.width, h0 = bitmap.height;
  const scale = RESIZE_SIZE / Math.min(w0, h0);
  const w1 = Math.round(w0 * scale), h1 = Math.round(h0 * scale);
  // Step 2: draw onto intermediate canvas, then center-crop to INPUT_SIZE
  const tmp = new OffscreenCanvas(w1, h1);
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(bitmap, 0, 0, w1, h1);
  const offX = Math.floor((w1 - INPUT_SIZE) / 2);
  const offY = Math.floor((h1 - INPUT_SIZE) / 2);
  const cropped = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const cctx = cropped.getContext('2d', { willReadFrequently: true });
  cctx.drawImage(tmp, offX, offY, INPUT_SIZE, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
  // Step 3: read pixels, normalize with CLIP stats, NCHW float32
  const { data } = cctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const f = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const p = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < p; i++) {
    f[0 * p + i] = (data[i * 4 + 0] / 255 - MEAN[0]) / STD[0];
    f[1 * p + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    f[2 * p + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', f, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}
