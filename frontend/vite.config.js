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
    proxy: {
      '/datasets': 'http://127.0.0.1:8000',
      '/analyses': 'http://127.0.0.1:8000',
      '/healthz': 'http://127.0.0.1:8000',
    },
  },
})
