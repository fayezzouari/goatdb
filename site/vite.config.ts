import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Relative asset paths, so the built site works from any directory on any static host.
export default defineConfig({
  base: './',
  plugins: [react()],
})
