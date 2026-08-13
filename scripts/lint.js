#!/usr/bin/env node
// scripts/lint.js — minimal sanity check for the extension sources.
// 1. Every .js file in extension/ parses as ESM or commonjs (node --check).
// 2. manifest.json is valid JSON and references files that exist.
// 3. No remote URLs in shipped code (privacy guarantee).

const fs = require('node:fs');
const path = require('node:path');
const { execSync } = require('node:child_process');

const ROOT = path.resolve(__dirname, '..');
const EXT = path.join(ROOT, 'extension');

let failed = false;
function fail(msg) { console.error('  FAIL:', msg); failed = true; }
function ok(msg)   { console.log('  OK:', msg); }

// 1. manifest.json
const manifestPath = path.join(EXT, 'manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
ok(`manifest.json parsed (v${manifest.version})`);

function mustExist(rel) {
  const p = path.join(EXT, rel);
  if (!fs.existsSync(p)) fail(`manifest references missing file: ${rel}`);
  else ok(`manifest ref exists: ${rel}`);
}

mustExist(manifest.background.service_worker);
for (const cs of manifest.content_scripts) {
  for (const j of cs.js) {
    if (j.startsWith('vendor/') && !fs.existsSync(path.join(EXT, j))) {
      // vendor scripts may be absent (fetched via fetch-model.sh) — warn.
      console.warn(`  WARN: vendor asset not yet present: ${j} (run scripts/fetch-model.sh)`);
    } else {
      mustExist(j);
    }
  }
  for (const c of cs.css || []) mustExist(c);
}

// 2. JS files parse
const jsFiles = [
  path.join(EXT, 'background/service-worker.js'),
  path.join(EXT, 'offscreen/offscreen.js'),
  path.join(EXT, 'content/content-script.js'),
  path.join(EXT, 'heuristics/heuristic-filter.js'),
  path.join(EXT, 'popup/popup.js'),
  path.join(ROOT, 'eval/runner.js'),
];
for (const f of jsFiles) {
  if (!fs.existsSync(f)) { fail(`source missing: ${path.relative(ROOT, f)}`); continue; }
  try {
    execSync(`node --check --input-type=module < ${JSON.stringify(f)}`, { stdio: 'pipe' });
  } catch {
    // node --check with input-type=module is finicky with imports; fall back to plain syntax check.
    try {
      execSync(`node --check "${f}"`, { stdio: 'pipe' });
      ok(`parse: ${path.relative(ROOT, f)}`);
    } catch (e) {
      fail(`parse error in ${path.relative(ROOT, f)}: ${e.message.split('\n')[0]}`);
    }
  }
}

// 3. Privacy: no remote URLs in shipped non-vendor code.
const remoteInCode = [];
function scan(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'vendor' || entry.name === 'node_modules') continue;
      scan(p);
    } else if (/\.(js|html|css)$/.test(entry.name)) {
      const content = fs.readFileSync(p, 'utf8');
      const m = content.match(/https?:\/\/[^\s'"\\)]+/g);
      if (m) {
        for (const url of m) {
          // Allow the maintainer link in the README / comments / docs.
          if (url.includes('poidh.xyz')) continue;
          if (url.includes('github.com')) continue;
          if (url.includes('npmjs.org')) continue;
          if (url.includes('kennyistyping')) continue;
          // chrome.runtime.getURL returns a chrome-extension:// URL — also OK.
          remoteInCode.push(`${path.relative(ROOT, p)}: ${url}`);
        }
      }
    }
  }
}
scan(EXT);
scan(path.join(ROOT, 'eval'));
if (remoteInCode.length) {
  fail(`remote URLs found in shipped code:
${remoteInCode.join('\n')}`);
} else {
  ok('no remote URLs in shipped code');
}

if (failed) {
  console.error('\nLINT FAILED');
  process.exit(1);
} else {
  console.log('\nLINT PASSED');
}
