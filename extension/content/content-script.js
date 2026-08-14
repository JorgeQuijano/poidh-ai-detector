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

  // Only score images that are plausibly "content". Icons, favicons,
  // avatars, buttons, sprites, and lazy-loaded thumbnails are skipped to
  // avoid badge noise. Tune freely:
  //   MIN_DISPLAY_PX  — rendered size on the page (both dimensions)
  //   MIN_INTRINSIC_PX — source file resolution (both dimensions)
  const MIN_DISPLAY_PX = 96;
  const MIN_INTRINSIC_PX = 128;

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
    // Skip SVG — createImageBitmap can't decode SVG, and the model isn't
    // trained on vector graphics anyway.
    if (src.toLowerCase().includes('.svg')) return null;

    let resp;
    try {
      resp = await fetch(src, { credentials: 'omit', cache: 'force-cache' });
    } catch {
      return null;
    }
    if (!resp.ok) return null;

    const blob = await resp.blob();
    if (blob.size < 1024) return null; // tiny placeholder / icon

    const buf = await blob.arrayBuffer();
    // Prefer the server-provided Content-Type. Fall back to inferring from
    // the URL extension so createImageBitmap can pick the right decoder.
    const mime = blob.type || guessMime(src);
    // Encode as base64 string. ArrayBuffer transfer across chrome.runtime
    // boundaries (content -> worker, worker -> offscreen via port) has been
    // observed to silently degrade to a plain object {0: ..., 1: ...} on
    // some Chromium builds, breaking image.bytes.byteLength downstream.
    // Base64 is structured-clone safe and survives any boundary.
    const bytes_b64 = arrayBufferToBase64(buf);
    return chrome.runtime.sendMessage({
      type: 'SCORE_IMAGE',
      image: {
        bytes_b64,
        mime,
        src,
      },
    });
  }

  // Convert an ArrayBuffer to a base64 string using chunked operations so
  // we don't blow up on large images (Wikipedia thumbs can be ~10MB).
  function arrayBufferToBase64(buf) {
    let binary = '';
    const bytes = new Uint8Array(buf);
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(binary);
  }

  // Map .png/.jpg/.gif/.webp/.svg URLs to a MIME type so we have something
  // for createImageBitmap to work with when the server omits Content-Type
  // (common on small CDN thumbnails and lazy-loaded images).
  function guessMime(url) {
    const u = url.split('?')[0].split('#')[0].toLowerCase();
    if (u.endsWith('.png'))  return 'image/png';
    if (u.endsWith('.jpg') || u.endsWith('.jpeg')) return 'image/jpeg';
    if (u.endsWith('.gif'))  return 'image/gif';
    if (u.endsWith('.webp')) return 'image/webp';
    if (u.endsWith('.avif')) return 'image/avif';
    if (u.endsWith('.bmp'))  return 'image/bmp';
    if (u.endsWith('.ico'))  return 'image/x-icon';
    return ''; // unknown — let createImageBitmap sniff
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
  // Decide whether an image is worth scoring. We gate on BOTH the size it's
  // rendered at on the page (display px) and its intrinsic resolution:
  //   - an icon rendered at 24px is noise even if the source file is 512px
  //   - a 20x20 source file is noise even if CSS stretches it to 400px
  function shouldScore(img) {
    const src = bestSrc(img);
    if (!src || src.startsWith('data:')) return false;   // inline data
    if (src.toLowerCase().includes('.svg')) return false; // vector, not decodable

    // Rendered size. 0 when hidden or not laid out yet — in that case fall
    // through to the intrinsic check below.
    const rect = img.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      if (rect.width < MIN_DISPLAY_PX || rect.height < MIN_DISPLAY_PX) return false;
    } else if (img.clientWidth > 0 && img.clientHeight > 0) {
      if (img.clientWidth < MIN_DISPLAY_PX || img.clientHeight < MIN_DISPLAY_PX) return false;
    }

    // Intrinsic resolution. 0 means not loaded / broken — skip those too.
    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
      if (img.naturalWidth < MIN_INTRINSIC_PX || img.naturalHeight < MIN_INTRINSIC_PX) return false;
    } else {
      return false;
    }
    return true;
  }

  async function processImg(img) {
    if (processed.has(img)) return;
    if (!img.isConnected) return;
    if (!shouldScore(img)) return; // skip icons / small / broken
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
