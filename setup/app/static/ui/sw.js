/**
 * Service worker for AeroSense.
 *
 * WHY THIS EXISTS AT ALL
 * Chrome will not offer to install an app unless a service worker with a
 * fetch handler is registered. Offline use is not a goal here - the app is
 * useless without its local FastAPI backend, which serves this very file -
 * so this worker stays deliberately small rather than trying to be clever.
 *
 * THE CACHING RULES, AND WHY
 *   /api/*, /ws  - never touched. Drill state, camera frames and player data
 *                  are live; a cached answer is a wrong answer.
 *   /assets/*    - cache-first. Vite content-hashes these filenames, so a
 *                  given URL's bytes never change and a rebuild produces new
 *                  URLs. Safe to keep forever.
 *   everything   - network-first, falling back to cache only when the network
 *   else         - fails. The document MUST NOT be cache-first: that is how a
 *                  PWA ends up serving yesterday's UI after scripts/build_ui.ps1
 *                  stages a new bundle.
 */

const CACHE = "aerosense-v1";

// Take over promptly so a rebuilt bundle is not shadowed by an old worker.
self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(names.filter((n) => n !== CACHE).map((n) => caches.delete(n)));
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only ever handle same-origin GETs. Cross-origin (e.g. the ESP32-CAM
  // stream on the LAN) and mutations go straight to the network.
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  // Live data: never cached, never intercepted.
  if (url.pathname.startsWith("/api/") || url.pathname === "/ws") return;

  if (url.pathname.startsWith("/assets/")) {
    event.respondWith(cacheFirst(request));
    return;
  }

  event.respondWith(networkFirst(request));
});

async function cacheFirst(request) {
  const hit = await caches.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) (await caches.open(CACHE)).put(request, res.clone());
  return res;
}

async function networkFirst(request) {
  try {
    const res = await fetch(request);
    if (res.ok) (await caches.open(CACHE)).put(request, res.clone());
    return res;
  } catch (err) {
    const hit = await caches.match(request);
    if (hit) return hit;
    throw err;
  }
}
