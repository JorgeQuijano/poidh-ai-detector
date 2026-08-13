// eval/onnxruntime-config.js
// Configure onnxruntime-web BEFORE the runner module loads. We pin to WASM
// and tell the runtime where to find the WASM binaries. The runtime resolves
// the WASM URLs against this base, so it must point at the directory that
// contains ort-wasm-simd-threaded.{wasm,mjs}.

window.ORT_CONFIG = {
  // Absolute-pathed so it survives whatever URL ort.min.js itself was loaded
  // from. The eval page lives at /eval/; the vendored runtime lives at
  // /extension/vendor/.
  wasmPaths: new URL('../extension/vendor/', document.baseURI).href,
  // We deliberately omit WebGPU so the eval matches the worst-case extension
  // environment.
  preferredExecutionProvider: 'wasm',
};
