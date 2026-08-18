import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built output is copied into src/api/static/ and served by FastAPI's
// StaticFiles mount at /ui (see src/api/app.py), so asset URLs must be
// relative rather than root-absolute.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: 'dist',
  },
})
