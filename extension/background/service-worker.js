// extension/background/service-worker.js
//
// MV3 service worker. Acts as the message router between the content script
// (which observes images) and the off-screen document (which hosts the
// onnxruntime-web WASM session — service workers cannot run WASM directly).
//
// Lifecycle:
//   - On first message that needs inference, ensure the offscreen document
//     exists and has loaded the model onnxruntime session.
//   - Heuristic-only requests (cheap metadata) are answered synchronously in
//     the worker without touching the offscreen doc.
//
// Privacy: image bytes flow content -> worker -> offscreen and return only
// labels. Nothing crosses the network boundary.

import { heuristicFilter } from '../heuristics/heuristic-filter.js';

const OFFSCREEN_URL = chrome.runtime.getURL('offscreen/offscreen.html');
const CONFIDENCE_THRESHOLD = 0.65;

// -- Offscreen document lifecycle --------------------------------------------

async function ensureOffscreen() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [OFFSCREEN_URL],
  });
  if (existing.length > 0) return existing[0].documentId || true;

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['WORKERS'],
    justification:
      'Run onnxruntime-web WASM inference for AI image detection. ' +
      'All processing stays on-device; no image data is sent off the machine.',
  });
  return true;
}

// -- Offscreen communication via a long-lived port -------------------------
// We use a dedicated port (not chrome.runtime.sendMessage) so the message
// is delivered to the offscreen document specifically, not echoed back to
// the service worker itself.

let offscreenPort = null;
let offscreenReadyPromise = null;

async function getOffscreenPort() {
  if (offscreenPort) return offscreenPort;
  if (offscreenReadyPromise) return offscreenReadyPromise;

  offscreenReadyPromise = (async () => {
    await ensureOffscreen();
    const port = chrome.runtime.connect({ name: 'poidh-offscreen' });
    // offscreen.js registers a port listener named 'poidh-offscreen'.
    offscreenPort = port;
    port.onDisconnect.addListener(() => {
      offscreenPort = null;
      offscreenReadyPromise = null;
    });
    return port;
  })();
  return offscreenReadyPromise;
}

function scoreViaOffscreen(image) {
  return new Promise(async (resolve, reject) => {
    try {
      const port = await getOffscreenPort();
      const requestId = Math.random().toString(36).slice(2);
      const listener = (msg) => {
        if (msg && msg.requestId === requestId) {
          port.onMessage.removeListener(listener);
          if (msg.ok) resolve(msg.result);
          else reject(new Error(msg.error || 'offscreen-failed'));
        }
      };
      port.onMessage.addListener(listener);
      port.postMessage({ requestId, image });
    } catch (err) {
      reject(err);
    }
  });
}

// -- Message routing --------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'PING') {
    sendResponse({ ok: true, version: chrome.runtime.getManifest().version });
    return false;
  }

  if (msg?.type === 'POIDH_DIAG') {
    // The offscreen doc doesn't have a devtools window users typically open,
    // so diagnostics are routed here and printed in the service worker console.
    console.warn('[poidh diag]', msg.tag, msg);
    return false;
  }

  if (msg?.type === 'SCORE_IMAGE') {
    // msg.image: { bitmap: ImageBitmap, src: string, width, height, mime, bytes? }
    handleScore(msg.image)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((err) => sendResponse({ ok: false, error: String(err && err.message || err) }));
    return true; // keep channel open for async response
  }

  return false;
});

async function handleScore(image) {
  // The content script sends only bytes (ArrayBuffer). ImageBitmap doesn't
  // survive structured cloning reliably across port.postMessage in some
  // Chromium builds, so we decode to a fresh ImageBitmap in the offscreen
  // document instead. The worker just forwards bytes.

  // Tier 1: heuristic pre-filter (cheap, no model load needed, works on bytes)
  const heuristic = await heuristicFilter(image);
  if (heuristic.confidence >= CONFIDENCE_THRESHOLD) {
    return {
      label: heuristic.label,                  // 'ai' | 'real'
      confidence: heuristic.confidence,
      tier: 'heuristic',
      reason: heuristic.reason,
    };
  }

  // Tier 2: neural inference via the offscreen document
  try {
    const neural = await scoreViaOffscreen(image);
    const { ai_probability, model_version } = neural;
    const label = ai_probability >= CONFIDENCE_THRESHOLD ? 'ai' : 'real';
    // Report confidence IN THE PREDICTED LABEL, not the raw AI probability:
    //   ai_probability 0.98 -> label 'ai'   -> confidence 0.98
    //   ai_probability 0.02 -> label 'real' -> confidence 0.98
    // Otherwise a "real" verdict would display as "real 2%" — reads like
    // we're 2% confident it's real when we're actually 98% confident.
    const confidence = label === 'ai' ? ai_probability : 1 - ai_probability;
    return {
      label,
      confidence,
      tier: 'neural',
      reason: `model=${model_version}`,
    };
  } catch (err) {
    // Offscreen failed: surface the error so the badge can show it.
    return {
      label: 'uncertain',
      confidence: heuristic.confidence,
      tier: 'heuristic',
      reason: `neural-failed: ${err.message || err}`,
    };
  }
}

// -- Install / startup --------------------------------------------------------

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.storage.local.set({
    installTime: Date.now(),
    stats: { scored: 0, ai: 0, real: 0, uncertain: 0 },
  });
});
