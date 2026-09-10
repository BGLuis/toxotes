// F0/F5: sob COEP:require-corp, carregar .wasm de CDN e bloqueado. A runtime do
// onnxruntime-web e do MediaPipe tem de ser servida da propria origem. Este script copia
// esses artefatos de node_modules/ para public/ (que o Vite serve na raiz).
//
// Roda automaticamente antes de `npm run dev` e `npm run build` (ver package.json).

import { cp, mkdir, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const at = (p) => join(root, p);

async function copyMatching(srcDir, destDir, keep) {
  if (!existsSync(srcDir)) {
    throw new Error(`nao encontrei ${srcDir} — rode "npm install" primeiro`);
  }
  await mkdir(destDir, { recursive: true });
  let n = 0;
  for (const name of await readdir(srcDir)) {
    if (!keep(name)) continue;
    await cp(join(srcDir, name), join(destDir, name));
    n += 1;
  }
  return n;
}

// Runtime WebAssembly do onnxruntime-web. ort.env.wasm.wasmPaths = '/ort/' aponta para ca.
// Só os builds que usamos: `.jsep` (caminho webgpu) e o simd-threaded puro (caminho wasm).
// Os builds `.asyncify` / `.jspi` nao entram na config default.
const ORT_KEEP = new Set([
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
]);
const ort = await copyMatching(
  at('node_modules/onnxruntime-web/dist'),
  at('public/ort'),
  (n) => ORT_KEEP.has(n),
);

// Runtime do MediaPipe: fetch em runtime por um caminho que o bundler nao enxerga.
// SIMD + o fallback nosimd; o build `_module_` e de um modo de threading que nao usamos.
const mp = await copyMatching(
  at('node_modules/@mediapipe/tasks-vision/wasm'),
  at('public/mediapipe/wasm'),
  (n) => n.startsWith('vision_wasm_internal.') || n.startsWith('vision_wasm_nosimd_internal.'),
);

console.log(`assets prontos: public/ort (${ort} arquivos), public/mediapipe/wasm (${mp} arquivos)`);
