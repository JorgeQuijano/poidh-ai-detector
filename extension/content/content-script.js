// extension/content/content-script.js
//
// Observes <img> (and <picture>) inserts via MutationObserver, fetches each
// image's bytes, runs the detect pipeline via the background service worker,
// and renders a small overlay badge {ai, real, uncertain} with confidence.
//
// Privacy: image bytes leave the page context only to the extension's own
// offscreen document, never to the network. The extension does not touch any
// image element the user disables via the popup toggle.

(() => {
  const BADGE_PREFIX = 'poidh-';
  const processed = new WeakSet();
  let enabled = true;
  let observer = null;

  // -- Config sync with popup ------------------------------------------------
  chrome.storage?.local?.get({ enabled: true }, (cfg) => {
    enabled = cfg.enabled !== false;
  });
  chrome.storage?.onChanged?.addListener((changes, area) => {
    if (area === 'local' && changes.enabled) {
      enabled = changes.enabled.newValue !== false;
      if (!enabled) removeAllBadges();
    }
  });

  // -- Image collection ------------------------------------------------------
  function collectImages(root) {
    if (!root || root.nodeType !== 1) return [];
    const out = [];
    const tag = root.tagName;
    if (tag === 'IMG') out.push(root);
    if (root.querySelectorAll) {
      out.push(...root.querySelectorAll('img'));
    }
    return out;
  }

  function bestSrc(img) {
    // Prefer the active source; fall back to src/srcset.
    if (img.currentSrc) return img.currentSrc;
    if (img.src) return img.src;
    if (img.srcset) {
      const first = img.srcset.split(',')[0].trim().split(' ')[0];
      if (first) return first;
    }
    return null;
  }

  // -- Inference ------------------------------------------------------------
  async function scoreImage(img) {
    const src = bestSrc(img);
    if (!src || src.startsWith('data:')) return null; // skip inline (too small)

    let resp;
    try {
      resp = await fetch(src, { credentials: 'omit', cache: 'force-cache' });
    } catch {
      return null;
    }
    if (!resp.ok) return null;

    const blob = await resp.blob();
    if (blob.size < 1024) return null; // tiny placeholder / icon

    let bitmap;
    try {
      bitmap = await createImageBitmap(blob);
    } catch {
      return null;
    }

    const buf = await blob.arrayBuffer();
    return chrome.runtime.sendMessage({
      type: 'SCORE_IMAGE',
      image: {
        bitmap,
        bytes: buf,
        mime: blob.type,
        src,
        width: bitmap.width,
        height: bitmap.height,
      },
    });
  }

  // -- Overlay badge --------------------------------------------------------
  function ensureBadgeContainer(img) {
    let host = img.parentElement;
    if (!host) return null;
    if (getComputedStyle(host).position === 'static') {
      host.style.position = 'relative';
    }
    let badge = host.querySelector(`:scope > .${BADGE_PREFIX}badge`);
    if (badge) return badge;
    badge = document.createElement('div');
    badge.className = `${BADGE_PREFIX}badge ${BADGE_PREFIX}pending`;
    badge.dataset.src = img.src || '';
    badge.textContent = '…';
    host.appendChild(badge);
    return badge;
  }

  function renderBadge(badge, result) {
    badge.classList.remove(`${BADGE_PREFIX}pending`, `${BADGE_PREFIX}ai`, `${BADGE_PREFIX}real`, `${BADGE_PREFIX}uncertain`);
    badge.classList.add(`${BADGE_PREFIX}${result.label}`);

    // If the service worker is reporting an uncertain verdict because the
    // neural tier failed (offscreen doc error, model load error, etc.), the
    // confidence is meaningless 0.0 — show the failure reason in the badge
    // body so the user can debug. Otherwise show label + confidence as usual.
    const isNeuralFailure =
      result.label === 'uncertain' &&
      result.tier === 'heuristic' &&
      typeof result.reason === 'string' &&
      result.reason.startsWith('neural-failed');

    if (isNeuralFailure) {
      const reason = result.reason.replace(/^neural-failed:\s*/, '');
      // Log full reason to console; show a truncated version in the badge.
      console.error('[poidh] neural inference failed:', reason);
      const short = reason.length > 32 ? reason.slice(0, 32) + '…' : reason;
      badge.textContent = `! ${short}`;
      badge.title = `poidh: neural inference failed — ${reason}`;
    } else if (result.label === 'uncertain' && (result.confidence ?? 0) < 0.001) {
      // Heuristic-only verdict with no positive evidence. Not a failure,
      // just "I don't know without the model" — keep the visual neutral.
      badge.textContent = `${result.label} —`;
      badge.title = `poidh: ${result.reason || 'no positive AI metadata'} — model would decide`;
    } else {
      badge.textContent = `${result.label} ${Math.round(result.confidence * 100)}%`;
      badge.title = `tier=${result.tier} • ${result.reason || ''}`;
    }
  }

  function removeAllBadges() {
    document.querySelectorAll(`.${BADGE_PREFIX}badge`).forEach((b) => b.remove());
  }

  // -- Per-image processing -------------------------------------------------
  async function processImg(img) {
    if (processed.has(img)) return;
    if (!img.isConnected) return;
    if (img.naturalWidth < 64 || img.naturalHeight < 64) return; // skip icons
    processed.add(img);

    const badge = ensureBadgeContainer(img);
    if (!badge) return;

    const reply = await scoreImage(img);
    if (!reply || !reply.ok) {
      badge.classList.remove(`${BADGE_PREFIX}pending`);
      badge.classList.add(`${BADGE_PREFIX}uncertain`);
      const reason = reply?.error || 'no-reply';
      // Show a short hint in the badge so it's obvious which failure mode
      // hit (heuristic crash, offscreen refused, model load error, etc.).
      const short = reason.length > 18 ? reason.slice(0, 18) + '…' : reason;
      badge.textContent = `? ${short}`;
      badge.title = `poidh: scoring failed — ${reason}`;
      return;
    }
    renderBadge(badge, reply.result);
  }

  // -- DOM observation ------------------------------------------------------
  function walkNew(root) {
    if (!enabled) return;
    for (const img of collectImages(root)) {
      if (img.complete) processImg(img);
      else img.addEventListener('load', () => processImg(img), { once: true });
    }
  }

  function startObserving() {
    if (observer) return;
    observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.addedNodes) {
          for (const node of m.addedNodes) {
            if (node.nodeType === 1) walkNew(node);
          }
        }
      }
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
    walkNew(document.documentElement);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startObserving, { once: true });
  } else {
    startObserving();
  }
})();
