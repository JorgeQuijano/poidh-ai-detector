// extension/offscreen/offscreen.js
//
// Runs inside the off-screen document. Loads the ONNX model (WASM-only — no
// WebGPU) and exposes a single message: 'OFFSCREEN_SCORE' -> { ai_probability }.
//
// Model is fetched from the extension's own package (chrome.runtime.getURL)
// and cached by the browser. We pin to WASM ep for worst-case Chrome compatibility.

const MODEL_URL = chrome.runtime.getURL('models/detector.int8.onnx');
const MODEL_VERSION = 'detector-int8-v0.1.0';
const INPUT_SIZE = 224; // square; ONNX model expects 224x224x3 normalized
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

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

// Bilinear downsample of an ImageBitmap into a CHW float32 tensor of shape
// [1, 3, 224, 224] normalized with ImageNet mean/std. This is the input
// contract for ResNet/EfficientNet/ViT-style backbones.
function bitmapToTensor(bitmap) {
  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixelCount = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < pixelCount; i++) {
    const r = data[i * 4 + 0] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    float32Data[0 * pixelCount + i] = (r - MEAN[0]) / STD[0];
    float32Data[1 * pixelCount + i] = (g - MEAN[1]) / STD[1];
    float32Data[2 * pixelCount + i] = (b - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

async function scoreImage(image) {
  const session = await getSession();
  const tensor = bitmapToTensor(image.bitmap);
  const feeds = { [session.inputNames[0]]: tensor };
  const out = await session.run(feeds);

  // Convention: model output is a single float [0..1] = P(ai). If the model
  // emits 2 logits instead, softmax + take class-1.
  const outTensor = out[session.outputNames[0]];
  let ai_probability;
  if (outTensor.dims.length === 1 && outTensor.dims[0] === 1) {
    ai_probability = outTensor.data[0];
  } else if (outTensor.dims.length === 1 && outTensor.dims[0] === 2) {
    const [a, b] = outTensor.data;
    const max = Math.max(a, b);
    const expA = Math.exp(a - max);
    const expB = Math.exp(b - max);
    ai_probability = expB / (expA + expB); // assume class 1 = ai
  } else {
    throw new Error(`unexpected model output shape: ${outTensor.dims}`);
  }
  return { ai_probability, model_version: MODEL_VERSION };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === 'OFFSCREEN_SCORE') {
    scoreImage(msg.image)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((err) => sendResponse({ ok: false, error: String(err && err.message || err) }));
    return true;
  }
  return false;
});
