const fs = require('fs');
const path = require('path');

const source = path.resolve(__dirname, '..', 'frontend', 'dist');
const target = path.resolve(__dirname, 'dist');

if (!fs.existsSync(source)) {
  console.error('Source dist not found:', source);
  process.exit(1);
}

if (fs.existsSync(target)) {
  fs.rmSync(target, { recursive: true, force: true });
}

fs.cpSync(source, target, { recursive: true });
console.log('Copied dist:', source, '->', target);
