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
  if (existing.length > 0) return;

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['WORKERS'],
    justification:
      'Run onnxruntime-web WASM inference for AI image detection. ' +
      'All processing stays on-device; no image data is sent off the machine.',
  });
}

// -- Message routing ---------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'PING') {
    sendResponse({ ok: true, version: chrome.runtime.getManifest().version });
    return false;
  }

  if (msg?.type === 'SCORE_IMAGE') {
    // msg.image: { bitmap: ImageBitmap, src: string, width, height, mime, bytes? }
    handleScore(msg.image)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep channel open for async response
  }

  return false;
});

async function handleScore(image) {
  // Tier 1: heuristic pre-filter (cheap, no model load needed)
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
  await ensureOffscreen();
  const neural = await chrome.runtime.sendMessage({
    type: 'OFFSCREEN_SCORE',
    image,
  });

  if (!neural?.ok) {
    return {
      label: 'uncertain',
      confidence: heuristic.confidence,
      tier: 'heuristic',
      reason: `neural-unavailable: ${neural?.error || 'unknown'}`,
    };
  }

  const { ai_probability, model_version } = neural.result;
  const label = ai_probability >= CONFIDENCE_THRESHOLD ? 'ai' : 'real';
  return {
    label,
    confidence: ai_probability,
    tier: 'neural',
    reason: `model=${model_version}`,
  };
}

// -- Install / startup --------------------------------------------------------

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.storage.local.set({
    installTime: Date.now(),
    stats: { scored: 0, ai: 0, real: 0, uncertain: 0 },
  });
});
