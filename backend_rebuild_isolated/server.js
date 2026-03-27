import cors from 'cors';
import express from 'express';
import fs from 'fs';
import fsp from 'fs/promises';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const RUNTIME_ROOT = process.pkg ? path.dirname(process.execPath) : __dirname;
const DATA_DIR = path.join(RUNTIME_ROOT, 'data');
const DATASET_DIR = path.join(DATA_DIR, 'dataset');
const LABELS_DIR = path.join(DATA_DIR, 'labels');
const MODELS_DIR = path.join(DATA_DIR, 'models');

async function ensureDirs() {
  await fsp.mkdir(DATASET_DIR, { recursive: true });
  await fsp.mkdir(LABELS_DIR, { recursive: true });
  await fsp.mkdir(MODELS_DIR, { recursive: true });
}

const storage = multer.diskStorage({
  destination: async (_req, _file, cb) => {
    await ensureDirs();
    cb(null, DATASET_DIR);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();
    const stem = path.basename(file.originalname || 'image', ext).replace(/[^a-zA-Z0-9-_]/g, '_');
    cb(null, `${Date.now()}_${stem}${ext || '.jpg'}`);
  }
});

const upload = multer({ storage });

function isImageName(name) {
  return /\.(jpg|jpeg|png|bmp|webp)$/i.test(name);
}

async function readLabelMap() {
  await ensureDirs();
  const names = await fsp.readdir(LABELS_DIR);
  const map = {};
  for (const name of names) {
    if (!name.endsWith('.json')) {
      continue;
    }
    const raw = await fsp.readFile(path.join(LABELS_DIR, name), 'utf-8');
    try {
      const payload = JSON.parse(raw);
      map[payload.fileName] = payload;
    } catch {
      // Ignore broken file and keep server available.
    }
  }
  return map;
}

async function listDataset() {
  await ensureDirs();
  const files = await fsp.readdir(DATASET_DIR);
  const labels = await readLabelMap();
  return files
    .filter(isImageName)
    .sort((a, b) => a.localeCompare(b))
    .map((fileName) => {
      const label = labels[fileName];
      return {
        fileName,
        url: `/api/files/${encodeURIComponent(fileName)}`,
        hasLabel: Boolean(label),
        score: label?.score ?? null
      };
    });
}

const app = express();
const port = process.env.PORT || 7766;

app.use(cors());
app.use(express.json({ limit: '30mb' }));
app.use('/api/files', express.static(DATASET_DIR));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, ts: Date.now() });
});

app.post('/api/dataset/upload-folder', upload.array('files', 500), async (_req, res) => {
  const dataset = await listDataset();
  res.json({ ok: true, files: dataset });
});

app.post('/api/dataset/import-existing', async (req, res) => {
  const { sourcePath } = req.body ?? {};
  if (!sourcePath) {
    res.status(400).json({ ok: false, message: 'sourcePath is required' });
    return;
  }

  try {
    await ensureDirs();
    const names = await fsp.readdir(sourcePath);
    for (const name of names) {
      if (!isImageName(name)) {
        continue;
      }
      const src = path.join(sourcePath, name);
      const dst = path.join(DATASET_DIR, name);
      try {
        await fsp.access(dst, fs.constants.F_OK);
      } catch {
        await fsp.copyFile(src, dst);
      }
    }
    const files = await listDataset();
    res.json({ ok: true, files });
  } catch (error) {
    res.status(500).json({ ok: false, message: String(error?.message || error) });
  }
});

app.get('/api/dataset/files', async (_req, res) => {
  const files = await listDataset();
  res.json({ ok: true, files });
});

app.post('/api/labels/:fileName', async (req, res) => {
  const fileName = req.params.fileName;
  const { score, keypoints, meta } = req.body ?? {};

  if (typeof score !== 'number') {
    res.status(400).json({ ok: false, message: 'score must be number' });
    return;
  }

  const payload = {
    fileName,
    score,
    keypoints: Array.isArray(keypoints) ? keypoints : [],
    meta: meta ?? {},
    updatedAt: new Date().toISOString()
  };

  await ensureDirs();
  const safeName = fileName.replace(/[^a-zA-Z0-9-_.]/g, '_');
  await fsp.writeFile(path.join(LABELS_DIR, `${safeName}.json`), JSON.stringify(payload, null, 2), 'utf-8');
  const files = await listDataset();
  const scored = files.filter((f) => f.hasLabel).length;
  res.json({ ok: true, progress: { scored, total: files.length } });
});

app.get('/api/labels', async (_req, res) => {
  const labels = await readLabelMap();
  res.json({ ok: true, labels });
});

app.get('/api/progress', async (_req, res) => {
  const files = await listDataset();
  const scored = files.filter((f) => f.hasLabel).length;
  res.json({ ok: true, scored, total: files.length });
});

app.post('/api/models', async (req, res) => {
  const { name, model } = req.body ?? {};
  if (!model || typeof model !== 'object') {
    res.status(400).json({ ok: false, message: 'model payload is required' });
    return;
  }

  await ensureDirs();
  const modelName = (name || `rf_model_${Date.now()}`).replace(/[^a-zA-Z0-9-_]/g, '_');
  const fileName = `${modelName}.json`;
  const payload = {
    ...model,
    savedAt: new Date().toISOString(),
    name: modelName
  };
  await fsp.writeFile(path.join(MODELS_DIR, fileName), JSON.stringify(payload, null, 2), 'utf-8');
  res.json({ ok: true, fileName, name: modelName });
});

app.get('/api/models', async (_req, res) => {
  await ensureDirs();
  const names = await fsp.readdir(MODELS_DIR);
  const models = names.filter((n) => n.endsWith('.json')).sort((a, b) => b.localeCompare(a));
  res.json({ ok: true, models });
});

app.get('/api/models/:fileName', async (req, res) => {
  const filePath = path.join(MODELS_DIR, req.params.fileName);
  try {
    const raw = await fsp.readFile(filePath, 'utf-8');
    res.json({ ok: true, model: JSON.parse(raw) });
  } catch {
    res.status(404).json({ ok: false, message: 'model not found' });
  }
});

app.post('/api/capture', upload.single('photo'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ ok: false, message: 'photo is required' });
    return;
  }
  const files = await listDataset();
  const captured = files.find((f) => f.fileName === req.file.filename) || null;
  res.json({ ok: true, captured, files });
});

const distCandidates = [
  path.join(__dirname, 'dist'),
  path.join(ROOT_DIR, 'frontend', 'dist'),
  path.join(path.dirname(process.execPath), 'dist')
];
const distDir = distCandidates.find((p) => fs.existsSync(p));
if (distDir) {
  app.use(express.static(distDir));
  app.get(/.*/, (_req, res) => {
    res.sendFile(path.join(distDir, 'index.html'));
  });
}

ensureDirs().then(() => {
  app.listen(port, () => {
    console.log(`Pose rebuild server running at http://localhost:${port}`);
  });
});
