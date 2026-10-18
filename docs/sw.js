/* SIGMA Service Worker: app shell + MODEL WEIGHTS are cached.
 * After first load the AI never re-downloads its brain — full offline. */
const SHELL = "sigma-shell-v1";
const MODELS = "sigma-models-v1";
const SHELL_ASSETS = ["./", "index.html", "style.css", "app.js", "ai.js", "games.js",
  "manifest.json", "ort.min.js", "ort-wasm-simd-threaded.wasm",
  "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(SHELL_ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys())
      if (k !== SHELL && k !== MODELS) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;

  // model weights + vocab: cache-first, never re-download
  if (url.pathname.includes("/models/") || url.pathname.endsWith("vocab.json")) {
    e.respondWith((async () => {
      const hit = await caches.open(MODELS).then(c => c.match(e.request));
      if (hit) return hit;
      const res = await fetch(e.request);
      if (res.ok) {
        const c = await caches.open(MODELS);
        c.put(e.request, res.clone());
      }
      return res;
    })());
    return;
  }

  // CDN (onnxruntime): cache-first too
  if (url.origin !== location.origin) {
    e.respondWith((async () => {
      const hit = await caches.open(SHELL).then(c => c.match(e.request));
      if (hit) return hit;
      const res = await fetch(e.request);
      if (res.ok) (await caches.open(SHELL)).put(e.request, res.clone());
      return res;
    })());
    return;
  }

  // app shell: cache-first with background refresh
  e.respondWith((async () => {
    const hit = await caches.match(e.request);
    const net = fetch(e.request).then(res => {
      if (res.ok) caches.open(SHELL).then(c => c.put(e.request, res.clone()));
      return res;
    }).catch(() => hit);
    return hit || net;
  })());
});
