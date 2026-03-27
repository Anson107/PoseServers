import { useEffect, useMemo, useRef, useState } from 'react';
import {
  API_BASE,
  capturePhoto,
  exportModelAsPklToFile,
  fetchDatasetFiles,
  fetchLabels,
  importExistingPath,
  listModels,
  loadModel,
  importPklModel,
  resetWorkspace,
  saveLabel,
  saveModel,
  uploadFolder
} from './api';
import { DEFAULT_IMPORT_PATH } from './constants';
import { detectSinglePose, drawPose, getFeatureNames, poseToFeatures } from './pose';
import { normalizeImportance, predictForest, scoreByRules, trainRandomForest } from './randomForest';
import { clamp, toPercent } from './utils';
import './App.css';

const TABS = [
  { key: 'annotation', label: '姿态标注' },
  { key: 'training', label: '模型训练' },
  { key: 'realtime', label: '实时评分' },
  { key: 'capture', label: '数据集采集' }
];

const BUILD_TAG = 'build-v23-20260326-reset-on-refresh';

function makeFeaturesFromKeypoints(keypoints) {
  return poseToFeatures({ keypoints })?.features ?? null;
}

function makePklFileName(baseName) {
  const clean = (baseName || 'archery_model').trim().replace(/[^a-zA-Z0-9-_]/g, '_') || 'archery_model';
  return `${clean}(1).pkl`;
}

function App() {
  const [activeTab, setActiveTab] = useState('annotation');
  const [datasetFiles, setDatasetFiles] = useState([]);
  const [labels, setLabels] = useState({});
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [status, setStatus] = useState('就绪');

  const [annotationScore, setAnnotationScore] = useState(50);
  const [annotationPose, setAnnotationPose] = useState(null);
  const [annotationFeatures, setAnnotationFeatures] = useState(null);
  const [sliderTouchedMap, setSliderTouchedMap] = useState({});

  const [trainParams, setTrainParams] = useState({ trees: 100, maxDepth: 10, minLeaf: 1 });
  const [trainedModel, setTrainedModel] = useState(null);
  const [modelNameInput, setModelNameInput] = useState('archery_rf_model');
  const [serverModels, setServerModels] = useState([]);
  const [selectedTree, setSelectedTree] = useState(0);

  const [rtMode, setRtMode] = useState('rf');
  const [rtScore, setRtScore] = useState(0);
  const [rtCameraOn, setRtCameraOn] = useState(false);
  const [rtHint, setRtHint] = useState('请导入模型并开启摄像头开始评分');
  const [rtReferenceName, setRtReferenceName] = useState('');
  const [rtReferenceFeatures, setRtReferenceFeatures] = useState(null);

  const [captureOn, setCaptureOn] = useState(false);
  const [captureHint, setCaptureHint] = useState('点击开启摄像头开始采集');

  const annotationImgRef = useRef(null);
  const annotationCanvasRef = useRef(null);

  const rtVideoRef = useRef(null);
  const rtCanvasRef = useRef(null);
  const rtStreamRef = useRef(null);
  const rtFrameRef = useRef(0);
  const rtBusyRef = useRef(false);
  const rtMissingPoseCountRef = useRef(0);
  const rtDetectCanvasRef = useRef(null);
  const rtLastInferTsRef = useRef(0);
  const rtLastPoseRef = useRef(null);
  const rtSmoothedScoreRef = useRef(null);

  const captureVideoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const captureStreamRef = useRef(null);

  const hiddenFolderInputRef = useRef(null);
  const hiddenModelInputRef = useRef(null);
  const hiddenRuleImageInputRef = useRef(null);

  const progress = useMemo(() => {
    const scored = datasetFiles.filter((f) => labels[f.fileName]).length;
    return { scored, total: datasetFiles.length };
  }, [datasetFiles, labels]);

  const currentFile = datasetFiles[selectedIndex] || null;
  const topImportance = useMemo(() => {
    if (!trainedModel?.importance) {
      return [];
    }
    return normalizeImportance(trainedModel.importance, getFeatureNames()).slice(0, 10);
  }, [trainedModel]);

  async function refreshAll() {
    const [files, labelMap, modelFiles] = await Promise.all([
      fetchDatasetFiles(),
      fetchLabels(),
      listModels()
    ]);
    setDatasetFiles(files);
    setLabels(labelMap);
    setServerModels(modelFiles);
  }

  useEffect(() => {
    setStatus('初始化中：正在重置为首次使用状态...');
    resetWorkspace()
      .then(() => refreshAll())
      .then(() => {
        setSelectedIndex(0);
        setTrainedModel(null);
        setRtReferenceFeatures(null);
        setRtReferenceName('');
        setStatus('已重置为首次使用状态');
      })
      .catch((err) => {
      setStatus(`初始化失败: ${String(err.message || err)}`);
      });
  }, []);

  useEffect(() => {
    if (!currentFile) {
      return;
    }
    const exist = labels[currentFile.fileName];
    setAnnotationScore(exist?.score ?? 50);
  }, [currentFile, labels]);

  function markCurrentAsScored(score) {
    if (!currentFile) {
      return;
    }
    setSliderTouchedMap((prev) => ({ ...prev, [currentFile.fileName]: true }));
    setLabels((prev) => ({
      ...prev,
      [currentFile.fileName]: {
        ...(prev[currentFile.fileName] || {}),
        fileName: currentFile.fileName,
        score,
        keypoints: annotationPose?.keypoints ?? prev[currentFile.fileName]?.keypoints ?? []
      }
    }));
  }

  async function saveScoreBySlider(score) {
    if (!currentFile) {
      return;
    }

    await saveLabel(currentFile.fileName, {
      score: clamp(score, 0, 100),
      keypoints: annotationPose?.keypoints ?? [],
      meta: {
        source: 'annotation-slider-auto-save',
        hasFeatures: Boolean(annotationFeatures)
      }
    });

    const labelMap = await fetchLabels();
    setLabels(labelMap);
    setStatus(`已打分: ${currentFile.fileName} = ${Math.round(score)}`);
  }

  useEffect(() => {
    return () => {
      stopRealtimeCamera();
      stopCaptureCamera();
    };
  }, []);

  async function onUploadFolder(evt) {
    const files = Array.from(evt.target.files || []);
    evt.target.value = '';
    if (!files.length) {
      return;
    }
    setStatus('正在上传文件夹...');
    const uploaded = await uploadFolder(files);
    setDatasetFiles(uploaded);
    const labelMap = await fetchLabels();
    setLabels(labelMap);
    setSelectedIndex(0);
    setStatus(`导入完成，共 ${uploaded.length} 张图片`);
  }

  async function onImportExisting() {
    const sourcePath = window.prompt('请输入数据源目录路径', DEFAULT_IMPORT_PATH) || '';
    if (!sourcePath.trim()) {
      return;
    }
    setStatus('正在导入已有数据目录...');
    const files = await importExistingPath(sourcePath.trim());
    setDatasetFiles(files);
    const labelMap = await fetchLabels();
    setLabels(labelMap);
    setSelectedIndex(0);
    setStatus(`导入完成，共 ${files.length} 张图片`);
  }

  async function detectOnAnnotationImage() {
    if (!annotationImgRef.current) {
      return;
    }
    setStatus('正在检测关键点...');
    try {
      const pose = await detectSinglePose(annotationImgRef.current, {
        selection: 'largest',
        mode: 'accurate'
      });
      setAnnotationPose(pose);
      drawPose(annotationCanvasRef.current, annotationImgRef.current, pose);
      const parsed = poseToFeatures(pose);
      setAnnotationFeatures(parsed?.features ?? null);
      setStatus(parsed ? '关键点检测完成' : '未检测到稳定姿态关键点，可手动打分后保存');
    } catch (err) {
      setAnnotationPose(null);
      setAnnotationFeatures(null);
      setStatus(`关键点检测失败: ${String(err.message || err)}`);
    }
  }

  async function onSaveLabel() {
    if (!currentFile) {
      return;
    }
    setStatus('正在保存标注...');
    await saveLabel(currentFile.fileName, {
      score: clamp(annotationScore, 0, 100),
      keypoints: annotationPose?.keypoints ?? [],
      meta: {
        source: 'annotation',
        hasFeatures: Boolean(annotationFeatures)
      }
    });
    const labelMap = await fetchLabels();
    setLabels(labelMap);
    setStatus(`已保存 ${currentFile.fileName} 的评分`);
  }

  async function onTrainModel() {
    const samples = [];
    const targets = [];

    for (const file of datasetFiles) {
      const label = labels[file.fileName];
      if (!label || typeof label.score !== 'number') {
        continue;
      }
      const features = makeFeaturesFromKeypoints(label.keypoints);
      if (!features) {
        continue;
      }
      samples.push(features);
      targets.push(label.score);
    }

    if (!samples.length) {
      setStatus('训练失败：未找到有效标注样本');
      return;
    }

    setStatus('正在训练模型，请稍候...');
    const model = trainRandomForest(samples, targets, {
      trees: trainParams.trees,
      maxDepth: trainParams.maxDepth,
      minLeaf: trainParams.minLeaf
    });
    setTrainedModel(model);
    setSelectedTree(0);
    setStatus(`训练完成，样本数 ${samples.length}`);
  }

  async function onExportModel() {
    if (!trainedModel) {
      return;
    }
    try {
      const payload = {
        ...trainedModel,
        featureNames: getFeatureNames()
      };
      const exportName = modelNameInput || 'archery_model';
      const result = await saveModel(exportName, payload);
      const downloadBase = makePklFileName(exportName).replace(/\.pkl$/i, '');

      // Use direct URL download to avoid WebView gesture restrictions on blob downloads.
      const downloadUrl = `${API_BASE}/api/models/${encodeURIComponent(result.fileName)}/export-pkl?downloadName=${encodeURIComponent(downloadBase)}&_t=${Date.now()}`;
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = makePklFileName(exportName);
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      a.remove();

      // Always export a filesystem copy for EXE environments where browser download can be blocked.
      const fallback = await exportModelAsPklToFile(result.fileName, downloadBase);

      const modelFiles = await listModels();
      setServerModels(modelFiles);
      setStatus(`模型已导出: ${result.fileName}，下载已触发；本地副本: ${fallback.savedPath}`);
    } catch (err) {
      setStatus(`模型导出失败: ${String(err?.message || err)}`);
    }
  }

  async function onImportModelFile(evt) {
    const file = evt.target.files?.[0];
    evt.target.value = '';
    if (!file) {
      return;
    }

    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (ext !== 'pkl') {
      setStatus(`模型导入失败: ${file.name}，当前仅支持 .pkl`);
      return;
    }

    try {
      setStatus(`正在解析 pkl: ${file.name}`);
      const { model, convertedName } = await importPklModel(file);
      setTrainedModel(model);
      const modelFiles = await listModels();
      setServerModels(modelFiles);
      setStatus(`pkl 模型已导入: ${convertedName || file.name}`);
    } catch (err) {
      setStatus(`模型导入失败: ${err?.message || file.name}`);
    }
  }

  async function onImportRuleReferenceImage(evt) {
    const file = evt.target.files?.[0];
    evt.target.value = '';
    if (!file) {
      return;
    }

    try {
      setRtHint(`正在解析参考图片: ${file.name}`);
      const localUrl = URL.createObjectURL(file);
      const img = new Image();
      img.src = localUrl;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      const pose = await detectSinglePose(img);
      const parsed = poseToFeatures(pose);
      URL.revokeObjectURL(localUrl);

      if (!parsed?.features) {
        setRtReferenceFeatures(null);
        setRtReferenceName('');
        setRtHint('参考图片未识别到有效人体姿态，请换一张人物清晰、全身可见的图片');
        return;
      }

      setRtReferenceFeatures(parsed.features);
      setRtReferenceName(file.name);
      setRtHint(`参考图片已加载: ${file.name}`);
    } catch (err) {
      setRtReferenceFeatures(null);
      setRtReferenceName('');
      setRtHint(`参考图片解析失败: ${String(err?.message || err)}`);
    }
  }

  async function onLoadServerModel(fileName) {
    const model = await loadModel(fileName);
    setTrainedModel(model);
    setStatus(`模型已加载: ${fileName}`);
  }

  async function startRealtimeCamera() {
    if (rtCameraOn) {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 540 },
        frameRate: { ideal: 30, max: 30 }
      }
    });
    rtStreamRef.current = stream;
    rtVideoRef.current.srcObject = stream;
    await rtVideoRef.current.play();
    setRtCameraOn(true);
    setRtHint('摄像头已开启，高精度姿态检测中...');
    rtMissingPoseCountRef.current = 0;
    rtLastInferTsRef.current = 0;
    rtLastPoseRef.current = null;
    rtSmoothedScoreRef.current = null;

    if (!rtDetectCanvasRef.current) {
      rtDetectCanvasRef.current = document.createElement('canvas');
    }
    rtDetectCanvasRef.current.width = 416;
    rtDetectCanvasRef.current.height = 234;

    const inferIntervalMs = 75;

    const loop = async () => {
      if (!rtStreamRef.current) {
        return;
      }

      const now = performance.now();

      if (!rtBusyRef.current && now - rtLastInferTsRef.current >= inferIntervalMs) {
        rtBusyRef.current = true;
        rtLastInferTsRef.current = now;
        try {
          const detectCanvas = rtDetectCanvasRef.current;
          const detectCtx = detectCanvas.getContext('2d');
          detectCtx.drawImage(rtVideoRef.current, 0, 0, detectCanvas.width, detectCanvas.height);

          const pose = await detectSinglePose(rtVideoRef.current, {
            detectionSource: detectCanvas,
            retrySource: rtVideoRef.current
          });
          rtLastPoseRef.current = pose;
          drawPose(rtCanvasRef.current, rtVideoRef.current, pose);

          const parsed = poseToFeatures(pose);
          if (parsed?.features) {
            rtMissingPoseCountRef.current = 0;
            let score = 0;
            if (rtMode === 'rf' && trainedModel?.trees?.length) {
              score = predictForest(trainedModel, parsed.features);
            } else {
              if (!rtReferenceFeatures?.length) {
                score = scoreByRules(parsed.features);
                setRtHint('规则评分中（当前未上传参考图，使用基础规则）');
              } else {
                score = scoreByRules(parsed.features, rtReferenceFeatures);
                setRtHint('规则评分中（已参考上传图片）');
              }
            }

            const safeScore = clamp(score, 0, 100);
            const alpha = 0.35;
            if (rtSmoothedScoreRef.current == null) {
              rtSmoothedScoreRef.current = safeScore;
            } else {
              rtSmoothedScoreRef.current = alpha * safeScore + (1 - alpha) * rtSmoothedScoreRef.current;
            }

            setRtScore(rtSmoothedScoreRef.current);
            if (rtMode === 'rf') {
              setRtHint('已检测到人体，随机森林实时评分中');
            }
          } else {
            rtMissingPoseCountRef.current += 1;
            if (rtMissingPoseCountRef.current > 12) {
              setRtHint('未检测到人体姿态，请调整站位/光照并确保人物在画面中');
            }
          }
        } catch (err) {
          setRtHint(`实时评分异常: ${String(err.message || err)}`);
        } finally {
          rtBusyRef.current = false;
        }
      } else if (rtLastPoseRef.current) {
        // Re-draw the latest pose between inference ticks to keep overlay responsive.
        drawPose(rtCanvasRef.current, rtVideoRef.current, rtLastPoseRef.current);
      }

      rtFrameRef.current = requestAnimationFrame(loop);
    };

    rtFrameRef.current = requestAnimationFrame(loop);
  }

  function stopRealtimeCamera() {
    cancelAnimationFrame(rtFrameRef.current);
    rtFrameRef.current = 0;
    if (rtStreamRef.current) {
      for (const track of rtStreamRef.current.getTracks()) {
        track.stop();
      }
    }
    rtStreamRef.current = null;
    setRtCameraOn(false);
    setRtHint('摄像头已关闭');
    rtMissingPoseCountRef.current = 0;
    rtLastPoseRef.current = null;
    rtSmoothedScoreRef.current = null;
  }

  function onScreenshotRealtime() {
    const video = rtVideoRef.current;
    if (!video) {
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (rtCanvasRef.current?.width) {
      ctx.drawImage(rtCanvasRef.current, 0, 0, canvas.width, canvas.height);
    }
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = `realtime_${Date.now()}.png`;
    a.click();
  }

  async function startCaptureCamera() {
    if (captureOn) {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    captureStreamRef.current = stream;
    captureVideoRef.current.srcObject = stream;
    await captureVideoRef.current.play();
    setCaptureOn(true);
    setCaptureHint('摄像头已开启，可以拍照采集');
  }

  function stopCaptureCamera() {
    if (captureStreamRef.current) {
      for (const track of captureStreamRef.current.getTracks()) {
        track.stop();
      }
    }
    captureStreamRef.current = null;
    setCaptureOn(false);
    setCaptureHint('摄像头已关闭');
  }

  async function onTakePhoto() {
    const video = captureVideoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas || !captureOn) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    if (!blob) {
      setCaptureHint('拍照失败，请重试');
      return;
    }

    setCaptureHint('正在保存照片到数据集...');
    const result = await capturePhoto(blob);
    setDatasetFiles(result.files || []);
    const labelMap = await fetchLabels();
    setLabels(labelMap);

    if (result.captured?.fileName) {
      const idx = (result.files || []).findIndex((f) => f.fileName === result.captured.fileName);
      if (idx >= 0) {
        setSelectedIndex(idx);
      }
    }
    setActiveTab('annotation');
    setCaptureHint('采集成功，已自动进入标注列表');
  }

  function renderTreeNode(node, depth = 0) {
    if (!node) {
      return null;
    }
    if (depth > 2) {
      return <div className="leaf-mini">...</div>;
    }
    if (node.type === 'leaf') {
      return <div className="leaf-mini">Score: {node.value.toFixed(2)}</div>;
    }
    return (
      <div className="tree-node-wrap">
        <div className="tree-node">
          f{node.feature} ≤ {node.threshold.toFixed(4)}
        </div>
        <div className="tree-children">
          <div>{renderTreeNode(node.left, depth + 1)}</div>
          <div>{renderTreeNode(node.right, depth + 1)}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">《我和AI一起学射箭》综合训练系统 ({BUILD_TAG})</div>
      </header>

      <div className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`tab ${activeTab === tab.key ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="status-row">
        <span>{status}</span>
        <span className="score-badge">
          已打分: {progress.scored} / {progress.total}
        </span>
      </div>

      <input
        ref={hiddenModelInputRef}
        type="file"
        accept=".pkl,application/octet-stream"
        style={{ display: 'none' }}
        onChange={onImportModelFile}
      />

      <input
        ref={hiddenRuleImageInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={onImportRuleReferenceImage}
      />

      {activeTab === 'annotation' && (
        <section className="annotation-layout">
          <aside className="left-list">
            <div className="toolbar">
              <button className="btn primary" onClick={() => hiddenFolderInputRef.current?.click()}>
                导入文件夹
              </button>
              <button className="btn" onClick={onImportExisting}>
                导入数据
              </button>
              <button className="btn success" onClick={onSaveLabel} disabled={!currentFile}>
                保存数据
              </button>
              <button className="btn" onClick={detectOnAnnotationImage} disabled={!currentFile}>
                重新检测
              </button>
              <input
                ref={hiddenFolderInputRef}
                type="file"
                accept="image/*"
                multiple
                style={{ display: 'none' }}
                onChange={onUploadFolder}
                webkitdirectory=""
                directory=""
              />
            </div>
            <div className="list-title">文件列表</div>
            <div className="file-list">
              {datasetFiles.map((f, idx) => (
                <button
                  key={f.fileName}
                  className={`file-item ${idx === selectedIndex ? 'active' : ''}`}
                  onClick={() => setSelectedIndex(idx)}
                >
                  {f.fileName}
                </button>
              ))}
            </div>
          </aside>

          <main className="annotation-main">
            <div className="image-stage">
              {currentFile ? (
                <div className="media-layer">
                  <img
                    ref={annotationImgRef}
                    src={`${API_BASE}${currentFile.url}`}
                    alt={currentFile.fileName}
                    onLoad={detectOnAnnotationImage}
                  />
                  <canvas ref={annotationCanvasRef} className="overlay" />
                </div>
              ) : (
                <div className="empty">开始您的标注工作</div>
              )}
            </div>

            <div className="panel">
              <div className="panel-title">标注控制台</div>
              <div className="label-row">
                <strong>姿态评分</strong>
                <span className="pill">人物 1</span>
                <span className="hint-inline">{annotationFeatures ? '已识别关键点' : '未识别关键点(可手动评分)'}</span>
              </div>
              <div className="slider-row">
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={annotationScore}
                  onChange={(e) => {
                    const score = Number(e.target.value);
                    setAnnotationScore(score);
                    markCurrentAsScored(score);
                  }}
                  onMouseUp={() => saveScoreBySlider(annotationScore)}
                  onTouchEnd={() => saveScoreBySlider(annotationScore)}
                />
                <div className="counter-box">{annotationScore}</div>
              </div>
              <div className="hint-inline">
                {currentFile && (sliderTouchedMap[currentFile.fileName] || labels[currentFile.fileName])
                  ? `当前已打分: ${Math.round(annotationScore)}`
                  : '默认值50，拖动滑条后即视为已打分'}
              </div>
              <div className="pager">
                <button className="btn" onClick={() => setSelectedIndex((v) => Math.max(0, v - 1))}>
                  上一张
                </button>
                <button
                  className="btn primary"
                  onClick={() => setSelectedIndex((v) => Math.min(datasetFiles.length - 1, v + 1))}
                >
                  下一张
                </button>
              </div>
            </div>
          </main>
        </section>
      )}

      {activeTab === 'training' && (
        <section className="train-layout">
          <div className="card">
            <h3>模型训练配置</h3>
            <div className="form-grid">
              <label>
                树的数量
                <input
                  type="number"
                  value={trainParams.trees}
                  min={10}
                  max={500}
                  onChange={(e) => setTrainParams((p) => ({ ...p, trees: Number(e.target.value) }))}
                />
              </label>
              <label>
                最大深度
                <input
                  type="number"
                  value={trainParams.maxDepth}
                  min={1}
                  max={12}
                  onChange={(e) => setTrainParams((p) => ({ ...p, maxDepth: Number(e.target.value) }))}
                />
              </label>
              <label>
                最小叶子样本
                <input
                  type="number"
                  value={trainParams.minLeaf}
                  min={1}
                  max={20}
                  onChange={(e) => setTrainParams((p) => ({ ...p, minLeaf: Number(e.target.value) }))}
                />
              </label>
            </div>
            <div className="row-actions">
              <button className="btn primary" onClick={onTrainModel}>
                开始训练
              </button>
              <button className="btn success" onClick={onExportModel} disabled={!trainedModel}>
                导出模型
              </button>
              <button className="btn warn" onClick={() => hiddenModelInputRef.current?.click()}>
                导入模型
              </button>
            </div>

            <div className="row-actions">
              <input
                className="name-input"
                value={modelNameInput}
                onChange={(e) => setModelNameInput(e.target.value)}
                placeholder="模型名称"
              />
              <select onChange={(e) => onLoadServerModel(e.target.value)} defaultValue="">
                <option value="" disabled>
                  加载服务器模型
                </option>
                {serverModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>

            {trainedModel && (
              <div className="metrics-box">
                <h4>模型性能</h4>
                <p>
                  MAE (平均/最小/最大): {trainedModel.metrics.mae.toFixed(4)} /{' '}
                  {trainedModel.metrics.minMae.toFixed(4)} / {trainedModel.metrics.maxMae.toFixed(4)}
                </p>
                <p>
                  深度 (平均/最大): {trainedModel.metrics.avgDepth.toFixed(1)} / {trainedModel.metrics.maxDepth}
                </p>
              </div>
            )}
          </div>

          <div className="card">
            <h3>特征重要性 (Top 10)</h3>
            <div className="importance-list">
              {topImportance.map((item) => (
                <div key={item.name} className="importance-item">
                  <div className="name">{item.name}</div>
                  <div className="bar-wrap">
                    <div className="bar" style={{ width: toPercent(item.score) }}></div>
                  </div>
                  <div className="num">{item.score.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>

          {trainedModel && (
            <>
              <div className="card forest-card">
                <h3>森林概览 ({trainedModel.params.trees})</h3>
                <div className="forest-list">
                  {trainedModel.treeMetrics.map((m) => (
                    <button
                      key={m.id}
                      className={`forest-item ${m.id === selectedTree ? 'active' : ''}`}
                      onClick={() => setSelectedTree(m.id)}
                    >
                      #{m.id} MAE: {m.mae.toFixed(3)}
                    </button>
                  ))}
                </div>
              </div>
              <div className="card tree-card">
                <h3>树 #{selectedTree} 结构</h3>
                {renderTreeNode(trainedModel.trees[selectedTree])}
              </div>
            </>
          )}
        </section>
      )}

      {activeTab === 'realtime' && (
        <section className="rt-layout">
          <aside className="card rt-controls">
            <h3>评分模式</h3>
            <div className="hint">当前构建: {BUILD_TAG}</div>
            <div className="mode-switch">
              <button
                className={`btn ${rtMode === 'rf' ? 'primary' : ''}`}
                onClick={() => setRtMode('rf')}
              >
                随机森林
              </button>
              <button
                className={`btn ${rtMode === 'rule' ? 'primary' : ''}`}
                onClick={() => setRtMode('rule')}
              >
                规则评分
              </button>
            </div>

            <h3>{rtMode === 'rf' ? '模型配置' : '参考配置'}</h3>
            <button
              className="btn"
              onClick={() => {
                if (rtMode === 'rf') {
                  hiddenModelInputRef.current?.click();
                } else {
                  hiddenRuleImageInputRef.current?.click();
                }
              }}
            >
              {rtMode === 'rf' ? '导入模型 (.pkl)' : '上传参考图片'}
            </button>
            {rtMode === 'rule' && (
              <div className="hint">{rtReferenceName ? `参考图片: ${rtReferenceName}` : '未上传参考图片'}</div>
            )}

            <h3>摄像头控制</h3>
            <div className="row-actions">
              <button className="btn success" onClick={startRealtimeCamera}>
                开启摄像头
              </button>
              <button className="btn danger" onClick={stopRealtimeCamera}>
                关闭摄像头
              </button>
            </div>
            <button className="btn" onClick={onScreenshotRealtime}>
              截图
            </button>
            <div className="live-score">当前评分: {rtScore.toFixed(1)}</div>
            <div className="hint">{rtHint}</div>
          </aside>

          <main className="card rt-stage">
            <div className="video-wrap">
              <div className="media-layer">
                <video ref={rtVideoRef} muted playsInline />
                <canvas ref={rtCanvasRef} className="overlay" />
              </div>
            </div>
          </main>
        </section>
      )}

      {activeTab === 'capture' && (
        <section className="capture-layout">
          <div className="card rt-controls">
            <h3>数据集采集</h3>
            <p>拍照后自动保存到数据集目录，并自动进入姿态标注列表。</p>
            <div className="row-actions">
              <button className="btn success" onClick={startCaptureCamera}>
                开启摄像头
              </button>
              <button className="btn danger" onClick={stopCaptureCamera}>
                关闭摄像头
              </button>
            </div>
            <button className="btn primary" onClick={onTakePhoto}>
              拍照采集
            </button>
            <div className="hint">{captureHint}</div>
            <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
          </div>
          <div className="card capture-stage">
            <video ref={captureVideoRef} muted playsInline />
          </div>
        </section>
      )}
    </div>
  );
}

export default App;
