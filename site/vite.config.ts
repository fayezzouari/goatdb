import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Served from https://fayezzouari.github.io/goatdb/
export default defineConfig({
  base: '/goatdb/',
  plugins: [react()],
})
