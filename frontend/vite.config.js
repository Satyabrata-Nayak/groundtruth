import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server runs on :5173 and the API on :8000 — two origins, so the browser
// would apply CORS to every request. The API does allow :5173 (see app/config.py),
// but proxying is better here for a reason that has nothing to do with convenience:
// with the proxy, the frontend calls same-origin relative paths like `/datasets`,
// exactly as it would if it were served from the API in production. Without it, the
// base URL becomes an environment variable, and "works in dev, 404s in prod" becomes
// a class of bug that exists.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // EVERY API PATH MUST BE LISTED HERE. A path that is missing does not 404 — Vite
    // falls back to serving index.html with a 200, so `res.json()` throws on an HTML
    // body and the failure surfaces as a component quietly rendering nothing. That is
    // exactly how the model picker shipped invisible: /models was added to the API and
    // not to this list, and a curl that only checked the status code said 200.
    //
    // `api.js` now detects an HTML body and says which line of this file to edit.
    proxy: {
      '/datasets': 'http://127.0.0.1:8000',
      '/analyses': 'http://127.0.0.1:8000',
      '/healthz': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
    },
  },
})
