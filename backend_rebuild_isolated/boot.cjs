// CJS bootstrap for pkg runtime to load ESM server entry.
import('./server.js').catch((err) => {
  console.error('Failed to start packaged server:', err);
  process.exit(1);
});
