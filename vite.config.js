import { createReadStream, existsSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { defineConfig, loadEnv } from 'vite';

// F0/F5: threads do WASM exigem SharedArrayBuffer -> exigem crossOriginIsolated -> exigem
// COOP/COEP. CORP: cross-origin evita que os proprios assets locais (servidos pelo Vite)
// sejam bloqueados sob require-corp.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'cross-origin',
};

// Tanto o onnxruntime-web (ort.env.wasm.wasmPaths = '/ort/') quanto o FilesetResolver do
// MediaPipe fazem `import()` de glue Emscripten (.js/.mjs) em runtime. O pipeline de
// dev/preview do Vite tenta transformar esses arquivos gerados e devolve 500/redireciona
// com `?import`. Este plugin serve /ort/* e /mediapipe/wasm/* crus, antes dos middlewares
// internos do Vite. Em produção os arquivos de public/ já vão crus para dist/.
const RAW_PREFIXES = ['/ort/', '/mediapipe/wasm/'];
const CONTENT_TYPE = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm' };

function serveRawRuntime() {
  const handler = (req, res, next) => {
    const url = req.url || '';
    if (!RAW_PREFIXES.some((p) => url.startsWith(p))) return next();
    const rel = normalize(url.split('?')[0]).replace(/^(\.\.[/\\])+/, '');
    const file = join(process.cwd(), 'public', rel);
    if (!existsSync(file)) return next();
    res.setHeader('Content-Type', CONTENT_TYPE[extname(file)] || 'application/octet-stream');
    for (const [k, v] of Object.entries(crossOriginIsolation)) res.setHeader(k, v);
    createReadStream(file).pipe(res);
  };
  return {
    name: 'serve-raw-runtime',
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const hostsEnv = env.ALLOWED_HOSTS || process.env.ALLOWED_HOSTS;

  let allowedHosts;
  if (hostsEnv === 'true') {
    allowedHosts = true;
  } else if (hostsEnv) {
    allowedHosts = hostsEnv.split(',').map((h) => h.trim()).filter(Boolean);
  } else {
    allowedHosts = ['luisarch', 'luisarch.dolly-kokanue.ts.net', '.ts.net'];
  }

  return {
    plugins: [serveRawRuntime()],
    server: {
      host: true,
      allowedHosts,
      headers: crossOriginIsolation,
    },
    preview: {
      host: true,
      allowedHosts,
      headers: crossOriginIsolation,
    },
    // Worker em modulo ES: so o onnxruntime-web roda dentro dele (a deteccao MediaPipe fica
    // na main thread — ver src/face-detect.js).
    worker: { format: 'es' },
    // onnxruntime-web traz .wasm/.mjs que o pre-bundle do Vite quebra; a runtime .wasm e
    // resolvida por import.meta.url e emitida em /assets pelo build.
    optimizeDeps: { exclude: ['onnxruntime-web'] },
    build: { target: 'es2022' },
  };
});
