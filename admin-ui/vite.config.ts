import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteSingleFile } from 'vite-plugin-singlefile'

// Bundle inline en un seul fichier HTML : `include_str!` côté Rust (voir
// `src/admin/ui.rs`) embarque ce fichier tel quel dans le binaire, servi
// same-origin par `/admin/panel` — aucun serveur de fichiers statiques à
// ajouter côté Axum, aucun souci de CORS.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    assetsInlineLimit: 100_000_000,
    cssCodeSplit: false,
    outDir: 'dist',
  },
  server: {
    port: 5184,
    // Dev uniquement : en prod ce build est servi same-origin PAR le service
    // Rust lui-même (voir `src/admin/ui.rs`), donc `fetch('/admin/...')` y
    // atteint directement l'API sans proxy. En dev, ce même chemin relatif
    // doit être redirigé vers le VPS — un proxy serveur-à-serveur contourne le
    // CORS du navigateur sans toucher à la configuration CORS du service Rust.
    // Le service n'écoute que sur 127.0.0.1 côté VPS (pas exposé publiquement) :
    // en dev, faire tourner un tunnel SSH local (`ssh -L 3002:127.0.0.1:3002 …`)
    // avant `npm run dev`, le proxy vise ensuite ce tunnel.
    proxy: {
      '/admin': 'http://localhost:3002',
    },
  },
})
