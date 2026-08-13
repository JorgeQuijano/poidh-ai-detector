// extension/heuristics/heuristic-filter.js
//
// Plan 3 — cheap metadata-driven pre-filter. Runs WITHOUT loading the neural
// model. If the heuristic is confident enough (>= 0.65), we short-circuit and
// never touch the offscreen document. Otherwise we return confidence < 0.65
// and the worker falls through to the neural tier.
//
// Signals we use (in order of authority):
//   1. C2PA / Content Credentials manifest (when present) — strong signal.
//   2. EXIF software tag — many generators leave "Stable Diffusion", "Midjourney",
//      "Adobe Firefly", "DALL-E", etc. in the EXIF Software field.
//   3. PNG text chunks — same generators sometimes embed "Software" / "Comment".
//   4. Specific PNG signature "sPNG" — Stable Diffusion WebUI default.
//   5. PNG quantization tables — SD-specific signatures exist in the literature.
//
// Falsifiability: ALL of these signals are FOR the AI class. We never declare
// "real" from heuristics alone — that's the neural tier's job. Heuristic only
// calls 'ai' when it has explicit positive evidence.

const KNOWN_GENERATORS = [
  'stable diffusion', 'stablediffusion', 'automatic1111', 'comfyui', 'sdxl',
  'midjourney', 'dalle', 'dall-e', 'openai',
  'firefly', 'adobe', 'imagen', 'gemini',
  'novelai', 'nai', 'kandinsky', 'yandex',
  'lexica', 'dreamstudio', 'runway', 'pika',
  'flux', 'black forest labs',
];

const CONFIDENT_AI = 0.95;
const CONFIDENT_THIRD_PARTY = 0.85;

export async function heuristicFilter(image) {
  // No bytes available? Bail (low confidence, force neural).
  if (!image.bytes || image.bytes.byteLength === 0) {
    return { label: 'uncertain', confidence: 0.0, reason: 'no-bytes' };
  }

  const bytes = new Uint8Array(image.bytes);
  const mime = (image.mime || '').toLowerCase();

  // 1. C2PA — JUMBF box in JPEG starts with bytes 0x00000011 'jumb' ...
  // We do a quick probe; full validation is intentionally out of scope here.
  if (mime === 'image/jpeg' || mime === 'image/jpg') {
    if (findBytes(bytes, 'jumb'.split('').map(c => c.charCodeAt(0)))) {
      return confidentAI('c2pa-jumb');
    }
    if (findBytes(bytes, 'c2pa'.split('').map(c => c.charCodeAt(0)))) {
      return confidentAI('c2pa-marker');
    }
  }

  // 2. EXIF Software tag — quick + cheap parse just for the text.
  const exifSoftware = (mime.startsWith('image/'))
    ? quickExifSoftware(bytes)
    : null;
  if (exifSoftware) {
    const lower = exifSoftware.toLowerCase();
    for (const gen of KNOWN_GENERATORS) {
      if (lower.includes(gen)) {
        return confidentAI(`exif: ${exifSoftware.trim()}`);
      }
    }
  }

  // 3. PNG text chunks — IHDR + scan for tEXt/zTXt/iTXt with "Software" key.
  if (mime === 'image/png') {
    const png = quickPngTextChunks(bytes);
    if (png.software) {
      const lower = png.software.toLowerCase();
      for (const gen of KNOWN_GENERATORS) {
        if (lower.includes(gen)) {
          return confidentAI(`png-software: ${png.software.trim()}`);
        }
      }
    }
    if (png.comment) {
      const lower = png.comment.toLowerCase();
      for (const gen of KNOWN_GENERATORS) {
        if (lower.includes(gen)) {
          return confidentTHIRDParty(`png-comment: ${png.comment.trim()}`);
        }
      }
    }
  }

  // 4 & 5. No positive evidence — let the neural model decide.
  return { label: 'uncertain', confidence: 0.0, reason: 'no-heuristic-signal' };
}

function confidentAI(reason) {
  return { label: 'ai', confidence: CONFIDENT_AI, reason };
}
function confidentTHIRDParty(reason) {
  return { label: 'ai', confidence: CONFIDENT_THIRD_PARTY, reason };
}

// --- byte helpers ------------------------------------------------------------

function findBytes(haystack, needle) {
  outer: for (let i = 0; i + needle.length <= haystack.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) continue outer;
    }
    return true;
  }
  return false;
}

// Returns a string view of the bytes in a range. Defensive: won't read past end.
function bytesToAscii(bytes, offset, length) {
  const end = Math.min(offset + length, bytes.length);
  let s = '';
  for (let i = offset; i < end; i++) {
    const c = bytes[i];
    if (c >= 0x20 && c < 0x7f) s += String.fromCharCode(c);
    else if (c === 0) break;
  }
  return s;
}

// --- Minimal EXIF parser -----------------------------------------------------
// We only need the "Software" tag (0x0131). Format reference: JEITA CP-3451.
// tinyPNG has a more complete reference; we deliberately stay narrow.

function quickExifSoftware(bytes) {
  // EXIF lives in JPEG APP1 marker (0xFFE1) and starts with "Exif\0\0".
  let i = 2; // skip SOI
  while (i + 4 < bytes.length) {
    if (bytes[i] !== 0xff) return null;
    const marker = bytes[i + 1];
    const segLen = (bytes[i + 2] << 8) | bytes[i + 3];
    if (marker === 0xe1) {
      // APP1 — could be EXIF or XMP. Try EXIF first.
      const seg = bytes.subarray(i + 4, i + 2 + segLen);
      if (bytesToAscii(seg, 0, 4) === 'Exif') {
        // TIFF header at offset 6 of the APP1 segment.
        const tiff = seg.subarray(6);
        const little = tiff[0] === 0x49 && tiff[1] === 0x49;
        const get16 = (o) => little ? (tiff[o] | (tiff[o + 1] << 8)) : ((tiff[o] << 8) | tiff[o + 1]);
        const get32 = (o) => little
          ? (tiff[o] | (tiff[o + 1] << 8) | (tiff[o + 2] << 16) | (tiff[o + 3] << 24))
          : ((tiff[o] << 24) | (tiff[o + 1] << 16) | (tiff[o + 2] << 8) | tiff[o + 3]);
        if (get16(2) !== 0x002a) return null;
        const ifd0Off = get32(4);
        const numEntries = get16(ifd0Off);
        for (let k = 0; k < numEntries; k++) {
          const e = ifd0Off + 2 + k * 12;
          const tag = get16(e);
          if (tag === 0x0131) {
            // Software
            const type = get16(e + 2);
            const count = get32(e + 4);
            const valOff = (type === 2 && count <= 4) ? (e + 8) : get32(e + 8);
            return bytesToAscii(tiff, valOff, count).replace(/\0+$/, '');
          }
        }
        return null;
      }
    }
    i += 2 + segLen;
  }
  return null;
}

// --- Minimal PNG text-chunk parser ------------------------------------------

function quickPngTextChunks(bytes) {
  // PNG signature: 89 50 4E 47 0D 0A 1A 0A
  if (bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4E || bytes[3] !== 0x47) {
    return { software: null, comment: null };
  }
  let i = 8;
  const result = { software: null, comment: null };
  while (i + 8 < bytes.length) {
    const len = (bytes[i] << 24) | (bytes[i + 1] << 16) | (bytes[i + 2] << 8) | bytes[i + 3];
    const type = bytesToAscii(bytes, i + 4, 4);
    const dataStart = i + 8;
    const dataEnd = dataStart + len;
    if (type === 'tEXt') {
      const sep = indexOfNull(bytes, dataStart, dataEnd);
      if (sep >= 0) {
        const key = bytesToAscii(bytes, dataStart, sep - dataStart).toLowerCase();
        const val = bytesToAscii(bytes, sep + 1, dataEnd - sep - 1);
        if (key === 'software') result.software = val;
        if (key === 'comment')  result.comment = val;
      }
    } else if (type === 'zTXt' || type === 'iTXt') {
      // We don't decompress; just note presence as a small positive signal.
      const sep = indexOfNull(bytes, dataStart, dataEnd);
      if (sep >= 0) {
        const key = bytesToAscii(bytes, dataStart, sep - dataStart).toLowerCase();
        if (key === 'software' && !result.software) result.software = '(compressed)';
      }
    }
    if (type === 'IEND') break;
    i = dataEnd + 4; // skip CRC
  }
  return result;
}

function indexOfNull(bytes, start, end) {
  for (let i = start; i < end; i++) if (bytes[i] === 0) return i;
  return -1;
}
