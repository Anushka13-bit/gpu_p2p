import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health':    { target: 'http://localhost:8000', changeOrigin: true },
      '/registry':  { target: 'http://localhost:8000', changeOrigin: true },
      '/progress':  { target: 'http://localhost:8000', changeOrigin: true },
      '/log':       { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
