// extension/popup/popup.js
const enabledEl = document.getElementById('enabled');
const scoredEl = document.getElementById('scored');
const aiEl = document.getElementById('ai');
const realEl = document.getElementById('real');
const uncertainEl = document.getElementById('uncertain');

chrome.storage.local.get({ enabled: true, stats: { scored: 0, ai: 0, real: 0, uncertain: 0 } }, (cfg) => {
  enabledEl.checked = cfg.enabled !== false;
  const s = cfg.stats || {};
  scoredEl.textContent = s.scored || 0;
  aiEl.textContent = s.ai || 0;
  realEl.textContent = s.real || 0;
  uncertainEl.textContent = s.uncertain || 0;
});

enabledEl.addEventListener('change', () => {
  chrome.storage.local.set({ enabled: enabledEl.checked });
});
