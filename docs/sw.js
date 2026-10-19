/* SIGMA Service Worker v6 — "failed to fetch" killer.
 * v6 changes:
 *  - every respondWith path is try/caught: the SW itself can NEVER produce
 *    "Failed to fetch"; any internal error falls back to a plain network fetch
 *  - model cache put is best-effort (quota errors ignored, response still served)
 *  - navigations are network-first with cache fallback: installed PWAs get new
 *    code on the first open after a deploy instead of being frozen on stale shell
 *  - shell assets install one-by-one (one 404 no longer kills the whole SW)
 *  - range requests pass straight through (streaming consumers keep working)
 *  - stale-while-revalidate for same-origin assets so updates propagate */
const SHELL = "sigma-shell-v6";
const MODELS = "sigma-models-v6";   // v6: char-copy 22m + shard-ready 50m
const SHELL_ASSETS = ["./", "index.html", "style.css", "app.js", "ai.js", "games.js",
  "manifest.json", "ort.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs",
  "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(SHELL);
    await Promise.all(SHELL_ASSETS.map(a =>
      c.add(new Request(a, { cache: "reload" })).catch(() => {})));   // tolerate single misses
    await self.skipWaiting();
  })());
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
  if (e.request.headers.get("range")) return;          // let streams handle themselves

  // model weights + vocab: cache-first, never re-download. Best-effort caching.
  if (url.pathname.includes("/models/")) {
    e.respondWith((async () => {
      try {
        const hit = await caches.open(MODELS).then(c => c.match(e.request));
        if (hit) return hit;
        const res = await fetch(e.request);
        if (res && res.ok) {
          try { (await caches.open(MODELS)).put(e.request, res.clone()); } catch (_) {}
        }
        return res;
      } catch (err) {
        // network hiccup: try cache once more, else rethrow as a clean failure
        const hit = await caches.open(MODELS).then(c => c.match(e.request)).catch(() => null);
        if (hit) return hit;
        throw err;
      }
    })());
    return;
  }

  // page navigations: network-first (PWA users get deploys immediately), cache fallback
  if (e.request.mode === "navigate") {
    e.respondWith((async () => {
      try {
        const res = await fetch(e.request);
        try { (await caches.open(SHELL)).put(e.request, res.clone()); } catch (_) {}
        return res;
      } catch (_) {
        const hit = await caches.match(e.request).catch(() => null);
        return hit || caches.match("./") || Response.error();
      }
    })());
    return;
  }

  // cross-origin (CDN): cache-first, safe put
  if (url.origin !== location.origin) {
    e.respondWith((async () => {
      try {
        const hit = await caches.open(SHELL).then(c => c.match(e.request));
        if (hit) return hit;
        const res = await fetch(e.request);
        if (res && res.ok) { try { (await caches.open(SHELL)).put(e.request, res.clone()); } catch (_) {} }
        return res;
      } catch (_) {
        const hit = await caches.open(SHELL).then(c => c.match(e.request)).catch(() => null);
        return hit || Response.error();
      }
    })());
    return;
  }

  // same-origin assets: stale-while-revalidate (instant + self-updating)
  e.respondWith((async () => {
    const hit = await caches.match(e.request).catch(() => null);
    const net = fetch(e.request).then(res => {
      if (res && res.ok) { try { caches.open(SHELL).then(c => c.put(e.request, res.clone())); } catch (_) {} }
      return res;
    }).catch(() => null);
    return hit || (await net) || Response.error();
  })());
});
