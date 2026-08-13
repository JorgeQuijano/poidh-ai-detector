// extension/offscreen/offscreen.js
//
// Runs inside the off-screen document. Loads the ONNX model (WASM-only — no
// WebGPU) and exposes a single port 'poidh-offscreen' that accepts
// { requestId, image } and replies with { requestId, ok, result } or
// { requestId, ok: false, error }.
//
// Model is fetched from the extension's own package (chrome.runtime.getURL)
// and cached by the browser. We pin to WASM ep for worst-case Chromium
// compatibility.

const MODEL_URL = chrome.runtime.getURL('models/detector-int8-v0.1.0.onnx');
const MODEL_VERSION = 'detector-int8-v0.1.0';

// CLIP ViT-L/14 input contract (matches UniversalFakeDetect validate.py):
//   1. shortest-side resize to 256
//   2. center-crop 224x224
//   3. CLIP normalization (NOT ImageNet)
const INPUT_SIZE = 224;
const RESIZE_SIZE = 256;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];

let sessionPromise = null;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      // Strip remote URLs / WebGPU / any attempt to phone home.
      ort.env.wasm.wasmPaths = chrome.runtime.getURL('vendor/');
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = 'warning';

      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],   // WASM-only by design
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
      });
      return session;
    })();
  }
  return sessionPromise;
}

// Preprocess an ImageBitmap to a CHW float32 tensor [1, 3, 224, 224] using
// the CLIP ViT-L/14 input contract (shortest-side 256, center-crop 224, CLIP stats).
function bitmapToTensor(bitmap) {
  // Step 1: shortest-side resize to RESIZE_SIZE.
  const w0 = bitmap.width, h0 = bitmap.height;
  const scale = RESIZE_SIZE / Math.min(w0, h0);
  const w1 = Math.round(w0 * scale), h1 = Math.round(h0 * scale);
  // Step 2: center-crop INPUT_SIZE.
  const tmp = new OffscreenCanvas(w1, h1);
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(bitmap, 0, 0, w1, h1);
  const offX = Math.floor((w1 - INPUT_SIZE) / 2);
  const offY = Math.floor((h1 - INPUT_SIZE) / 2);
  const cropped = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const cctx = cropped.getContext('2d', { willReadFrequently: true });
  cctx.drawImage(tmp, offX, offY, INPUT_SIZE, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
  // Step 3: read pixels, normalize with CLIP stats, NCHW float32.
  const { data } = cctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixelCount = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < pixelCount; i++) {
    float32Data[0 * pixelCount + i] = (data[i * 4 + 0] / 255 - MEAN[0]) / STD[0];
    float32Data[1 * pixelCount + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    float32Data[2 * pixelCount + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

async function scoreImage(image) {
  const session = await getSession();
  const tensor = bitmapToTensor(image.bitmap);
  const feeds = { [session.inputNames[0]]: tensor };
  const out = await session.run(feeds);

  // Model output is 2 logits [real, fake] (our adapted probe). Take the
  // softmax probability of the "fake" class.
  const outTensor = out[session.outputNames[0]];
  let ai_probability;
  if (outTensor.dims.length === 1 && outTensor.dims[0] === 1) {
    ai_probability = outTensor.data[0];
  } else if (outTensor.dims.length === 1 && outTensor.dims[0] === 2) {
    const [a, b] = outTensor.data;
    const max = Math.max(a, b);
    const expA = Math.exp(a - max);
    const expB = Math.exp(b - max);
    ai_probability = expB / (expA + expB); // index 1 = "fake"
  } else {
    throw new Error(`unexpected model output shape: ${outTensor.dims}`);
  }
  return { ai_probability, model_version: MODEL_VERSION };
}

// Listen on the dedicated port 'poidh-offscreen'. The service worker
// connects via chrome.runtime.connect({ name: 'poidh-offscreen' }).
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'poidh-offscreen') return;
  port.onMessage.addListener(async (msg) => {
    const { requestId, image } = msg || {};
    if (!requestId) return;
    try {
      const result = await scoreImage(image);
      port.postMessage({ requestId, ok: true, result });
    } catch (err) {
      console.error('[poidh offscreen] scoreImage failed:', err);
      port.postMessage({ requestId, ok: false, error: String(err && err.message || err) });
    }
  });
});
