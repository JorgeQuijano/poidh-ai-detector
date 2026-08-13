// eval/runner.js
//
// Loads onnxruntime-web, runs the same pipeline as the extension's offscreen
// document over a folder of images, and reports balanced accuracy at the
// maintained threshold (default 0.65).
//
// THIS FILE IS THE LOCAL REGRESSION TEST. It mirrors what we expect the
// bounty maintainer to do: load a balanced set of real + AI images, score
// each, and aggregate. A local pass at >75% is necessary (not sufficient)
// before claiming the bounty.

import { heuristicFilter } from '../extension/heuristics/heuristic-filter.js';

const MODEL_URL = '../extension/models/detector-int8-v0.1.0.onnx';
const MODEL_VERSION = 'detector-int8-v0.1.0';
const INPUT_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session = null;

async function loadOrt() {
  // ort.min.js is a UMD bundle that exposes `window.ort`. Load it as a
  // classic script so the global is set, then return it.
  if (window.ort) return window.ort;
  await new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = '../extension/vendor/ort.min.js';
    s.onload = () => resolve();
    s.onerror = () => reject(new Error('failed to load ort.min.js'));
    document.head.appendChild(s);
  });
  if (window.ORT_CONFIG) {
    ort.env.wasm.wasmPaths = window.ORT_CONFIG.wasmPaths;
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'warning';
  }
  return ort;
}

async function getSession() {
  if (session) return session;
  await loadOrt();
  // Probe whether the model file exists. If not, surface a clear error so
  // the user knows to run scripts/fetch-model.sh.
  try {
    const head = await fetch(MODEL_URL, { method: 'HEAD' });
    if (!head.ok) {
      throw new Error(`model not found at ${MODEL_URL} (HTTP ${head.status}). Run scripts/fetch-model.sh or drop a 224x224 ONNX int8 model at extension/models/detector-int8-v0.1.0.onnx.`);
    }
  } catch (err) {
    throw new Error(`model not found: ${err.message}`);
  }
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return session;
}

function bitmapToTensor(bitmap) {
  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixelCount = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < pixelCount; i++) {
    float32Data[0 * pixelCount + i] = (data[i * 4 + 0] / 255 - MEAN[0]) / STD[0];
    float32Data[1 * pixelCount + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    float32Data[2 * pixelCount + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

async function scoreImage(file) {
  const buf = await file.arrayBuffer();
  const blob = new Blob([buf], { type: file.type });
  const bitmap = await createImageBitmap(blob);
  const heuristic = await heuristicFilter({
    bitmap,
    bytes: buf,
    mime: file.type,
    src: file.name,
    width: bitmap.width,
    height: bitmap.height,
  });
  if (heuristic.confidence >= parseFloat(document.getElementById('threshold').value)) {
    return { label: heuristic.label, confidence: heuristic.confidence, tier: 'heuristic', reason: heuristic.reason };
  }
  // Heuristic alone was not enough — fall through to the neural tier.
  let s;
  try {
    s = await getSession();
  } catch (err) {
    return { label: 'uncertain', confidence: heuristic.confidence, tier: 'heuristic', reason: `model-missing: ${err.message}` };
  }
  const tensor = bitmapToTensor(bitmap);
  const out = await s.run({ [s.inputNames[0]]: tensor });
  const t = out[s.outputNames[0]];
  let ai_probability;
  if (t.dims.length === 1 && t.dims[0] === 1) {
    ai_probability = t.data[0];
  } else if (t.dims.length === 1 && t.dims[0] === 2) {
    const [a, b] = t.data;
    const max = Math.max(a, b);
    ai_probability = Math.exp(b - max) / (Math.exp(a - max) + Math.exp(b - max));
  }
  return {
    label: ai_probability >= parseFloat(document.getElementById('threshold').value) ? 'ai' : 'real',
    confidence: ai_probability,
    tier: 'neural',
    reason: MODEL_VERSION,
  };
}

function setStatus(text) {
  document.getElementById('status').textContent = text;
}
function setProgress(pct) {
  document.getElementById('bar').style.width = `${(pct * 100).toFixed(1)}%`;
}

async function run(files) {
  if (!files?.length) {
    setStatus('no files selected');
    return;
  }

  // The bountry maintainer won't tell us which class a file is — we infer
  // from the parent folder name. Two classes: ai/ and real/.
  const items = [...files].map((f) => ({
    file: f,
    truth: f.webkitRelativePath?.toLowerCase().includes('/ai/') ? 'ai' : 'real',
  }));

  const threshold = parseFloat(document.getElementById('threshold').value);
  const results = [];
  for (let i = 0; i < items.length; i++) {
    const { file, truth } = items[i];
    setStatus(`scoring ${i + 1}/${items.length} — ${file.name}`);
    setProgress((i + 1) / items.length);
    try {
      const r = await scoreImage(file);
      results.push({ file: file.name, truth, ...r });
    } catch (err) {
      results.push({ file: file.name, truth, error: String(err && err.message || err) });
    }
  }

  report(results, threshold);
}

function report(results, threshold) {
  // Per-class confusion
  const cm = {
    ai:   { tp: 0, fn: 0, fp: 0, tn: 0 },
    real: { tp: 0, fn: 0, fp: 0, tn: 0 },
  };
  for (const r of results) {
    if (r.error) continue;
    const pred = r.label;
    if (r.truth === 'ai') {
      if (pred === 'ai') cm.ai.tp++; else cm.ai.fn++;
      if (pred === 'real') cm.real.fp++; else cm.real.tn++;
    } else {
      if (pred === 'real') cm.real.tp++; else cm.real.fn++;
      if (pred === 'ai') cm.ai.fp++; else cm.ai.tn++;
    }
  }

  const tprAi = (cm.ai.tp + cm.ai.fn) ? cm.ai.tp / (cm.ai.tp + cm.ai.fn) : 0;
  const tprReal = (cm.real.tp + cm.real.fn) ? cm.real.tp / (cm.real.tp + cm.real.fn) : 0;
  const balanced = (tprAi + tprReal) / 2;

  const totalAi = cm.ai.tp + cm.ai.fn;
  const totalReal = cm.real.tp + cm.real.fn;
  const accAi = totalAi ? cm.ai.tp / totalAi : 0;
  const accReal = totalReal ? cm.real.tp / totalReal : 0;

  document.getElementById('results').style.display = '';
  document.getElementById('row-ai').innerHTML =
    `<td>ai</td><td>${totalAi}</td><td>${cm.ai.tp} / ${cm.ai.fn}</td>` +
    `<td>${cm.ai.fp} / ${cm.ai.tn}</td><td>${tprAi.toFixed(3)}</td>` +
    `<td>${totalReal ? (cm.ai.fp / totalReal).toFixed(3) : '—'}</td>` +
    `<td>${accAi.toFixed(3)}</td>`;
  document.getElementById('row-real').innerHTML =
    `<td>real</td><td>${totalReal}</td><td>${cm.real.tp} / ${cm.real.fn}</td>` +
    `<td>${cm.real.fp} / ${cm.real.tn}</td><td>${tprReal.toFixed(3)}</td>` +
    `<td>${totalAi ? (cm.real.fp / totalAi).toFixed(3) : '—'}</td>` +
    `<td>${accReal.toFixed(3)}</td>`;

  const balancedCell = document.getElementById('balanced-acc');
  balancedCell.textContent = `${(balanced * 100).toFixed(2)}% @ threshold ${threshold}`;
  balancedCell.className = balanced >= 0.75 ? 'pass' : 'fail';

  const verdict = document.getElementById('verdict');
  verdict.style.display = '';
  if (balanced >= 0.75) {
    verdict.className = 'verdict pass';
    verdict.textContent = `PASS — balanced accuracy ${(balanced * 100).toFixed(2)}% meets the 75% bar.`;
  } else {
    verdict.className = 'verdict fail';
    verdict.textContent = `FAIL — balanced accuracy ${(balanced * 100).toFixed(2)}% < 75% bar.`;
  }

  document.getElementById('per-image').textContent = JSON.stringify(results, null, 2);
  setStatus(`done — ${results.length} images scored at threshold ${threshold}`);
}

document.getElementById('run').addEventListener('click', () => {
  const files = document.getElementById('folder').files;
  run(files);
});

document.getElementById('demo').addEventListener('click', async () => {
  // Smoke test: build a 3-image dataset from the canvas (no real eval data).
  // Useful for verifying the pipeline runs end-to-end without any model.
  const synthetic = [];
  for (let i = 0; i < 3; i++) {
    const c = new OffscreenCanvas(64, 64);
    const ctx = c.getContext('2d');
    ctx.fillStyle = ['#ff6b6b', '#4ecdc4', '#ffe66d'][i];
    ctx.fillRect(0, 0, 64, 64);
    const blob = await c.convertToBlob({ type: 'image/png' });
    synthetic.push(new File([blob], `synthetic-${i}.png`, { type: 'image/png' }));
  }
  // No webkitRelativePath — we'll treat them all as 'real' for the demo.
  for (const f of synthetic) Object.defineProperty(f, 'webkitRelativePath', { value: 'demo/real/' + f.name });
  setStatus('running demo (no model expected; heuristic-only path)');
  run(synthetic);
});
