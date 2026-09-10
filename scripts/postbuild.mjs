// O Rollup emite uma copia da runtime .wasm do onnxruntime-web em dist/assets/ (por causa
// de um `new URL(..., import.meta.url)` interno da lib), mas em runtime o ORT busca de
// /ort (ort.env.wasm.wasmPaths). Removemos a copia orfa de dist/assets/ — a versao servida
// e dist/ort/, vinda de public/ort/.

import { readdir, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const assetsDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'assets');

let removed = 0;
for (const name of await readdir(assetsDir)) {
  if (name.startsWith('ort-wasm-') && name.endsWith('.wasm')) {
    await rm(join(assetsDir, name));
    removed += 1;
  }
}
console.log(`postbuild: ${removed} runtime .wasm orfa(s) removida(s) de dist/assets/`);
