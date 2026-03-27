import axios from 'axios';

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:7766';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000
});

export async function fetchDatasetFiles() {
  const { data } = await client.get('/api/dataset/files');
  return data.files || [];
}

export async function resetWorkspace() {
  const { data } = await client.post('/api/reset-workspace');
  return data;
}

export async function uploadFolder(files) {
  const form = new FormData();
  for (const file of files) {
    form.append('files', file);
  }
  const { data } = await client.post('/api/dataset/upload-folder', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return data.files || [];
}

export async function importExistingPath(sourcePath) {
  const { data } = await client.post('/api/dataset/import-existing', { sourcePath });
  return data.files || [];
}

export async function fetchLabels() {
  const { data } = await client.get('/api/labels');
  return data.labels || {};
}

export async function saveLabel(fileName, payload) {
  const { data } = await client.post(`/api/labels/${encodeURIComponent(fileName)}`, payload);
  return data;
}

export async function saveModel(name, model) {
  const { data } = await client.post('/api/models', { name, model });
  return data;
}

export async function exportModelAsPkl(fileName, downloadName) {
  const { data } = await client.get(
    `/api/models/${encodeURIComponent(fileName)}/export-pkl`,
    {
      params: { downloadName },
      responseType: 'blob',
      timeout: 120000
    }
  );
  return data;
}

export async function exportModelAsPklToFile(fileName, downloadName) {
  const { data } = await client.post(
    `/api/models/${encodeURIComponent(fileName)}/export-pkl-file`,
    { downloadName },
    { timeout: 120000 }
  );
  return data;
}

export async function listModels() {
  const { data } = await client.get('/api/models');
  return data.models || [];
}

export async function loadModel(fileName) {
  const { data } = await client.get(`/api/models/${encodeURIComponent(fileName)}`);
  return data.model;
}

export async function importPklModel(file) {
  const form = new FormData();
  form.append('modelFile', file);
  try {
    const { data } = await client.post('/api/models/import-pkl', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000
    });
    return data;
  } catch (err) {
    const message = err?.response?.data?.message || err?.message || 'pkl导入失败';
    throw new Error(message);
  }
}

export async function capturePhoto(blob) {
  const form = new FormData();
  form.append('photo', blob, `capture_${Date.now()}.jpg`);
  const { data } = await client.post('/api/capture', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return data;
}
