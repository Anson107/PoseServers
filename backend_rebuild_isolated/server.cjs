const cors = require('cors');
const express = require('express');
const fs = require('fs');
const fsp = require('fs/promises');
const multer = require('multer');
const path = require('path');
const os = require('os');
const { spawn, spawnSync } = require('child_process');

const ROOT_DIR = path.resolve(__dirname, '..');
const RUNTIME_ROOT = process.pkg ? path.dirname(process.execPath) : __dirname;
const DATA_DIR = path.join(RUNTIME_ROOT, 'data');
const DATASET_DIR = path.join(DATA_DIR, 'dataset');
const LABELS_DIR = path.join(DATA_DIR, 'labels');
const MODELS_DIR = path.join(DATA_DIR, 'models');
const EXPORTS_DIR = path.join(RUNTIME_ROOT, 'exports');

async function ensureDirs() {
  await fsp.mkdir(DATASET_DIR, { recursive: true });
  await fsp.mkdir(LABELS_DIR, { recursive: true });
  await fsp.mkdir(MODELS_DIR, { recursive: true });
  await fsp.mkdir(EXPORTS_DIR, { recursive: true });
}

async function clearDir(dirPath) {
  await fsp.mkdir(dirPath, { recursive: true });
  const names = await fsp.readdir(dirPath);
  await Promise.all(
    names.map((name) => fsp.rm(path.join(dirPath, name), { recursive: true, force: true }))
  );
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
const uploadModel = multer({ storage: multer.memoryStorage() });

function resolvePythonScriptPath(scriptRelativePath, tempDirForRuntime) {
  const embeddedScriptPath = path.join(__dirname, scriptRelativePath);
  if (!process.pkg) {
    return embeddedScriptPath;
  }

  const scriptContent = fs.readFileSync(embeddedScriptPath, 'utf-8');
  const tempScriptPath = path.join(
    tempDirForRuntime,
    `${path.basename(scriptRelativePath, '.py')}_runtime.py`
  );
  fs.writeFileSync(tempScriptPath, scriptContent, 'utf-8');
  return tempScriptPath;
}

function runPythonConverter(pklPath, jsonPath) {
  const scriptPath = resolvePythonScriptPath(path.join('scripts', 'convert_sklearn_pkl.py'), path.dirname(pklPath));

  const attempts = [
    { cmd: process.env.PYTHON || 'python', args: [scriptPath, pklPath, jsonPath] },
    { cmd: 'py', args: ['-3', scriptPath, pklPath, jsonPath] },
    { cmd: 'python3', args: [scriptPath, pklPath, jsonPath] }
  ];

  let lastError = '';
  for (const item of attempts) {
    const res = spawnSync(item.cmd, item.args, {
      encoding: 'utf-8',
      timeout: 120000,
      windowsHide: true
    });
    if (!res.error && res.status === 0) {
      return;
    }
    lastError = `${item.cmd}: ${res.stderr || res.error?.message || `exit ${res.status}`}`;
  }

  throw new Error(`Python转换失败。请检查 Python 环境与依赖（scikit-learn）。${lastError}`);
}

function runPythonJsonToPklConverter(jsonPath, pklPath) {
  const scriptPath = resolvePythonScriptPath(path.join('scripts', 'convert_json_to_pickle.py'), path.dirname(pklPath));

  const attempts = [
    { cmd: process.env.PYTHON || 'python', args: [scriptPath, jsonPath, pklPath] },
    { cmd: 'py', args: ['-3', scriptPath, jsonPath, pklPath] },
    { cmd: 'python3', args: [scriptPath, jsonPath, pklPath] }
  ];

  let lastError = '';
  for (const item of attempts) {
    const res = spawnSync(item.cmd, item.args, {
      encoding: 'utf-8',
      timeout: 120000,
      windowsHide: true
    });
    if (!res.error && res.status === 0) {
      return;
    }
    lastError = `${item.cmd}: ${res.stderr || res.error?.message || `exit ${res.status}`}`;
  }

  throw new Error(`Python导出PKL失败。请检查 Python 环境。${lastError}`);
}

function isImageName(name) {
  return /\.(jpg|jpeg|png|bmp|webp)$/i.test(name);
}

function tryParseJsonModelFromBuffer(buffer) {
  if (!buffer || !buffer.length) {
    return null;
  }
  const text = buffer.toString('utf-8').trim();
  if (!text.startsWith('{') || !text.endsWith('}')) {
    return null;
  }
  try {
    const parsed = JSON.parse(text);
    if (parsed && Array.isArray(parsed.trees)) {
      return parsed;
    }
  } catch {
    // Not a valid JSON model payload.
  }
  return null;
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
      // ignore bad label file
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
        score: label ? label.score : null
      };
    });
}

const app = express();
const port = process.env.PORT || 7766;

function openUrlInDefaultBrowser(url) {
  try {
    if (process.platform === 'win32') {
      const child = spawn('cmd', ['/c', 'start', '', url], {
        detached: true,
        stdio: 'ignore',
        windowsHide: true
      });
      child.unref();
      return;
    }

    if (process.platform === 'darwin') {
      const child = spawn('open', [url], { detached: true, stdio: 'ignore' });
      child.unref();
      return;
    }

    const child = spawn('xdg-open', [url], { detached: true, stdio: 'ignore' });
    child.unref();
  } catch {
    // ignore auto-open errors
  }
}

app.use(cors());
app.use(express.json({ limit: '30mb' }));
app.use('/api/files', express.static(DATASET_DIR));

// Force browser refresh to always fetch newest frontend bundle.
app.use((req, res, next) => {
  if (req.method === 'GET' && !req.path.startsWith('/api/')) {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.setHeader('Surrogate-Control', 'no-store');
  }
  next();
});

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, ts: Date.now() });
});

app.post('/api/reset-workspace', async (_req, res) => {
  try {
    await ensureDirs();
    await Promise.all([
      clearDir(DATASET_DIR),
      clearDir(LABELS_DIR),
      clearDir(MODELS_DIR),
      clearDir(EXPORTS_DIR)
    ]);
    res.json({ ok: true });
  } catch (error) {
    res.status(500).json({ ok: false, message: String((error && error.message) || error) });
  }
});

app.post('/api/dataset/upload-folder', upload.array('files', 500), async (_req, res) => {
  const dataset = await listDataset();
  res.json({ ok: true, files: dataset });
});

app.post('/api/dataset/import-existing', async (req, res) => {
  const sourcePath = req.body && req.body.sourcePath;
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
    res.status(500).json({ ok: false, message: String((error && error.message) || error) });
  }
});

app.get('/api/dataset/files', async (_req, res) => {
  const files = await listDataset();
  res.json({ ok: true, files });
});

app.post('/api/labels/:fileName', async (req, res) => {
  const fileName = req.params.fileName;
  const score = req.body && req.body.score;
  const keypoints = req.body && req.body.keypoints;
  const meta = req.body && req.body.meta;

  if (typeof score !== 'number') {
    res.status(400).json({ ok: false, message: 'score must be number' });
    return;
  }

  const payload = {
    fileName,
    score,
    keypoints: Array.isArray(keypoints) ? keypoints : [],
    meta: meta || {},
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
  const name = req.body && req.body.name;
  const model = req.body && req.body.model;
  if (!model || typeof model !== 'object') {
    res.status(400).json({ ok: false, message: 'model payload is required' });
    return;
  }

  await ensureDirs();
  const modelName = (name || `rf_model_${Date.now()}`).replace(/[^a-zA-Z0-9-_]/g, '_');
  const fileName = `${modelName}.json`;
  const payload = Object.assign({}, model, {
    savedAt: new Date().toISOString(),
    name: modelName
  });
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

app.get('/api/models/:fileName/export-pkl', async (req, res) => {
  const fileName = req.params.fileName;
  if (!fileName.endsWith('.json')) {
    res.status(400).json({ ok: false, message: '仅支持导出已保存的json模型' });
    return;
  }

  const modelPath = path.join(MODELS_DIR, fileName);
  const suggested = (req.query.downloadName || path.basename(fileName, '.json'))
    .toString()
    .replace(/[^a-zA-Z0-9-_]/g, '_') || 'archery_model';

  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'pose-export-pkl-'));
  const tempJson = path.join(tempDir, 'model.json');
  const tempPkl = path.join(tempDir, `${suggested}.pkl`);

  try {
    const raw = await fsp.readFile(modelPath, 'utf-8');
    JSON.parse(raw);
    await fsp.writeFile(tempJson, raw, 'utf-8');

    runPythonJsonToPklConverter(tempJson, tempPkl);
    const bin = await fsp.readFile(tempPkl);

    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${suggested}.pkl"`);
    res.send(bin);
  } catch (error) {
    res.status(500).json({ ok: false, message: String((error && error.message) || error) });
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
  }
});

app.post('/api/models/:fileName/export-pkl-file', async (req, res) => {
  const fileName = req.params.fileName;
  if (!fileName.endsWith('.json')) {
    res.status(400).json({ ok: false, message: '仅支持导出已保存的json模型' });
    return;
  }

  const modelPath = path.join(MODELS_DIR, fileName);
  const suggested = (req.body?.downloadName || path.basename(fileName, '.json'))
    .toString()
    .replace(/[^a-zA-Z0-9-_]/g, '_') || 'archery_model';

  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'pose-export-pkl-file-'));
  const tempJson = path.join(tempDir, 'model.json');
  const tempPkl = path.join(tempDir, `${suggested}.pkl`);
  const finalPath = path.join(EXPORTS_DIR, `${suggested}_${Date.now()}.pkl`);

  try {
    const raw = await fsp.readFile(modelPath, 'utf-8');
    JSON.parse(raw);
    await fsp.writeFile(tempJson, raw, 'utf-8');

    runPythonJsonToPklConverter(tempJson, tempPkl);
    await fsp.copyFile(tempPkl, finalPath);

    res.json({ ok: true, savedPath: finalPath, fileName: path.basename(finalPath) });
  } catch (error) {
    res.status(500).json({ ok: false, message: String((error && error.message) || error) });
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
  }
});

app.post('/api/models/import-pkl', uploadModel.single('modelFile'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ ok: false, message: 'modelFile is required' });
    return;
  }

  const ext = path.extname(req.file.originalname || '').toLowerCase();
  if (ext !== '.pkl') {
    res.status(400).json({ ok: false, message: '当前仅支持 .pkl 模型导入' });
    return;
  }

  await ensureDirs();
  const original = req.file.originalname || `archery_model_${Date.now()}.pkl`;
  const stem = path.basename(original, path.extname(original)).replace(/[^a-zA-Z0-9-_]/g, '_');
  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'pose-pkl-'));
  const pklPath = path.join(tempDir, `${stem}.pkl`);
  const convertedPath = path.join(tempDir, `${stem}.json`);

  try {
    let model = tryParseJsonModelFromBuffer(req.file.buffer);

    if (!model) {
      await fsp.writeFile(pklPath, req.file.buffer);
      runPythonConverter(pklPath, convertedPath);
      const raw = await fsp.readFile(convertedPath, 'utf-8');
      model = JSON.parse(raw);
    }

    const modelName = `${stem}_imported_${Date.now()}`;
    const savedName = `${modelName}.json`;
    const payload = Object.assign({}, model, {
      name: modelName,
      savedAt: new Date().toISOString(),
      source: model?.source || 'pkl-import'
    });

    await fsp.writeFile(path.join(MODELS_DIR, savedName), JSON.stringify(payload, null, 2), 'utf-8');
    res.json({ ok: true, model: payload, convertedName: savedName });
  } catch (error) {
    res.status(500).json({ ok: false, message: String((error && error.message) || error) });
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
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
  app.use(
    express.static(distDir, {
      etag: false,
      maxAge: 0,
      setHeaders: (res) => {
        res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
        res.setHeader('Pragma', 'no-cache');
        res.setHeader('Expires', '0');
      }
    })
  );
  app.get(/.*/, (_req, res) => {
    res.sendFile(path.join(distDir, 'index.html'));
  });
}

ensureDirs().then(() => {
  app.listen(port, () => {
    const url = `http://localhost:${port}`;
    console.log(`Pose rebuild server running at ${url}`);

    // In packaged EXE mode, open the app in the system default browser automatically.
    if (process.pkg && process.env.NO_AUTO_OPEN !== '1') {
      setTimeout(() => openUrlInDefaultBrowser(url), 500);
    }
  });
});
