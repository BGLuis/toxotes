// F6 — cache-first para os artefatos pesados (modelos + runtime .wasm). Sem isto, cada
// visita re-baixa dezenas de MB. Nao intercepta mais nada.
const CACHE = 'fer-assets-v1';
const PREFIXES = ['/models/', '/mediapipe/', '/ort/', '/assets/'];

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (!PREFIXES.some((p) => url.pathname.startsWith(p))) return;

  event.respondWith(
    caches.open(CACHE).then(async (cache) => {
      const hit = await cache.match(event.request);
      if (hit) return hit;
      const res = await fetch(event.request);
      if (res.ok) cache.put(event.request, res.clone());
      return res;
    }),
  );
});
