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

// Decode image bytes into a fresh ImageBitmap. Handles the case where the
// caller already provided an ImageBitmap (skip decode) or only bytes.
async function decodeBitmap(image) {
  if (image.bitmap && typeof image.bitmap.width === 'number' && image.bitmap.width > 0) {
    return image.bitmap;
  }
  if (!image.bytes) {
    throw new Error('image has neither bitmap nor bytes');
  }
  // Try with the provided MIME first; fall back to sniffing (no type) so
  // createImageBitmap reads the magic bytes. This handles servers that
  // omit Content-Type or send the wrong one.
  const tryDecoders = [
    () => createImageBitmap(new Blob([image.bytes], { type: image.mime || 'application/octet-stream' })),
    () => createImageBitmap(new Blob([image.bytes])), // sniff
  ];
  let lastErr;
  for (const fn of tryDecoders) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
    }
  }
  // Diagnostic: surface what we actually got so the next bug is debuggable.
  const u8 = new Uint8Array(image.bytes, 0, Math.min(8, image.bytes.byteLength));
  const magic = Array.from(u8).map(b => b.toString(16).padStart(2, '0')).join(' ');
  console.error(
    '[poidh offscreen] decode failed',
    { src: image.src, mime: image.mime, size: image.bytes.byteLength, magic }
  );
  // Also send to the service worker so it shows up in the SW devtools,
  // which is the only console most users open.
  try {
    chrome.runtime.sendMessage({
      type: 'POIDH_DIAG',
      tag: 'decode-failed',
      src: image.src,
      mime: image.mime,
      size: image.bytes.byteLength,
      magic,
    });
  } catch {}
  throw new Error(`decode failed [magic=${magic} size=${image.bytes.byteLength} mime=${image.mime} src=${image.src}]`);
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      // Strip remote URLs / WebGPU / any attempt to phone home.
      // The WASM loader needs explicit URLs for both the JS-side wrapper
      // (.mjs) and the wasm binary. The default fallback would resolve
      // against document.baseURI which lands in the wrong directory inside
      // the offscreen document, so we pass both explicitly.
      const base = chrome.runtime.getURL('vendor/');
      ort.env.wasm.wasmPaths = {
        mjs: new URL('ort-wasm-simd-threaded.jsep.mjs', base).href,
        wasm: new URL('ort-wasm-simd-threaded.jsep.wasm', base).href,
      };
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
  // Defensive: a tiny / corrupt / not-yet-decoded bitmap can report zero
  // dimensions, which would give Math.min(0,0)=0 and 256/0=Infinity. The
  // OffscreenCanvas constructor rejects NaN/Infinity/0 widths.
  if (!Number.isFinite(w0) || !Number.isFinite(h0) || w0 < 1 || h0 < 1) {
    throw new Error(`invalid bitmap dimensions: ${w0}x${h0}`);
  }
  const scale = RESIZE_SIZE / Math.min(w0, h0);
  // Math.round can still produce 0 if the source is degenerate.
  const w1 = Math.max(1, Math.round(w0 * scale));
  const h1 = Math.max(1, Math.round(h0 * scale));
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
  // Decode the bytes into a fresh ImageBitmap here. The content script
  // sends only ArrayBuffer (ImageBitmap doesn't survive structured
  // cloning across port.postMessage in some Chromium builds).
  const bitmap = await decodeBitmap(image);
  const tensor = bitmapToTensor(bitmap);
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
