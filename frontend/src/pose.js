import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import * as posenet from '@tensorflow-models/posenet';

const detectorCache = new Map();
const FEATURE_MIN_SCORE = 0.1;
const DRAW_MIN_SCORE = 0.06;
const MIN_VISIBLE_FEATURE_POINTS = 6;

const KEYPOINT_NAMES = [
  'nose',
  'leftEye',
  'rightEye',
  'leftEar',
  'rightEar',
  'leftShoulder',
  'rightShoulder',
  'leftElbow',
  'rightElbow',
  'leftWrist',
  'rightWrist',
  'leftHip',
  'rightHip',
  'leftKnee',
  'rightKnee',
  'leftAnkle',
  'rightAnkle'
];

function pickBackend() {
  return tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
}

function normalizeName(name = '') {
  return name.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
}

function poseVisibleKeypoints(pose, minScore = FEATURE_MIN_SCORE) {
  return (pose?.keypoints || []).filter((p) => (p?.score ?? 0) >= minScore);
}

function poseBoxOnSource(pose, minScore = FEATURE_MIN_SCORE) {
  const vis = poseVisibleKeypoints(pose, minScore);
  if (!vis.length) {
    return null;
  }
  const xs = vis.map((p) => p.position.x);
  const ys = vis.map((p) => p.position.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return {
    minX,
    minY,
    maxX,
    maxY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY),
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2
  };
}

function chooseBestPose(poses, source, options = {}) {
  if (!poses?.length) {
    return null;
  }

  const srcW = source.videoWidth || source.naturalWidth || source.width || source.clientWidth || 1;
  const srcH = source.videoHeight || source.naturalHeight || source.height || source.clientHeight || 1;
  const srcArea = Math.max(1, srcW * srcH);
  const targetMode = options.selection || 'center';

  let best = null;
  let bestScore = -Infinity;

  for (const pose of poses) {
    const vis = poseVisibleKeypoints(pose, FEATURE_MIN_SCORE);
    if (vis.length < MIN_VISIBLE_FEATURE_POINTS) {
      continue;
    }

    const box = poseBoxOnSource(pose, FEATURE_MIN_SCORE);
    if (!box) {
      continue;
    }

    const avgConf = vis.reduce((acc, p) => acc + (p.score || 0), 0) / vis.length;
    const areaRatio = Math.min(1, (box.width * box.height) / srcArea);
    const dx = Math.abs(box.cx / srcW - 0.5);
    const dy = Math.abs(box.cy / srcH - 0.5);
    const centerScore = Math.max(0, 1 - Math.min(1, (dx + dy) * 1.6));

    const finalScore =
      targetMode === 'largest'
        ? areaRatio * 0.6 + avgConf * 0.4
        : avgConf * 0.5 + centerScore * 0.35 + areaRatio * 0.15;

    if (finalScore > bestScore) {
      bestScore = finalScore;
      best = pose;
    }
  }

  return best || poses[0];
}

export async function getDetector(mode = 'fast') {
  const key = mode === 'accurate' ? 'accurate' : 'fast';
  if (!detectorCache.has(key)) {
    const promise = (async () => {
      await pickBackend();
      await tf.ready();
      if (key === 'accurate') {
        return posenet.load({
          architecture: 'ResNet50',
          outputStride: 16,
          inputResolution: { width: 513, height: 513 },
          quantBytes: 2
        });
      }

      return posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: 353, height: 353 },
        multiplier: 0.75,
        quantBytes: 2
      });
    })();
    detectorCache.set(key, promise);
  }
  return detectorCache.get(key);
}

function mapPoseToTarget(pose, source, mediaElement) {
  const srcW = source.videoWidth || source.naturalWidth || source.width || source.clientWidth || 1;
  const srcH = source.videoHeight || source.naturalHeight || source.height || source.clientHeight || 1;
  const dstW = mediaElement.videoWidth || mediaElement.naturalWidth || mediaElement.clientWidth || srcW;
  const dstH = mediaElement.videoHeight || mediaElement.naturalHeight || mediaElement.clientHeight || srcH;
  const sx = dstW / srcW;
  const sy = dstH / srcH;

  return {
    keypoints: pose.keypoints.map((p) => ({
      x: p.position.x * sx,
      y: p.position.y * sy,
      score: p.score,
      name: normalizeName(p.part || '')
    }))
  };
}

function confidentCount(pose) {
  if (!pose?.keypoints?.length) {
    return 0;
  }
  return pose.keypoints.filter((p) => (p.score ?? 0) >= FEATURE_MIN_SCORE).length;
}

export async function detectSinglePose(mediaElement, options = {}) {
  const detector = await getDetector(options.mode || 'fast');
  const source = options.detectionSource || mediaElement;
  let pose = null;

  const candidates = await detector.estimateMultiplePoses(source, {
    flipHorizontal: false,
    maxDetections: 5,
    scoreThreshold: 0.08,
    nmsRadius: 20
  });
  pose = chooseBestPose(candidates, source, options);

  if (!pose) {
    pose = await detector.estimateSinglePose(source, {
      flipHorizontal: false
    });
  }

  // Retry on original media when low-res detection misses upper-body keypoints.
  const retrySource = options.retrySource || (source !== mediaElement ? mediaElement : null);
  if (retrySource && confidentCount(pose) < MIN_VISIBLE_FEATURE_POINTS) {
    const retryCandidates = await detector.estimateMultiplePoses(retrySource, {
      flipHorizontal: false,
      maxDetections: 5,
      scoreThreshold: 0.08,
      nmsRadius: 20
    });
    const retryPose =
      chooseBestPose(retryCandidates, retrySource, options) ||
      (await detector.estimateSinglePose(retrySource, {
        flipHorizontal: false
      }));
    if (confidentCount(retryPose) > confidentCount(pose)) {
      pose = retryPose;
      return mapPoseToTarget(pose, retrySource, mediaElement);
    }
  }

  if (!pose?.keypoints?.length) {
    return null;
  }
  return mapPoseToTarget(pose, source, mediaElement);
}

function getSourceSize(mediaElement) {
  return {
    width: mediaElement.videoWidth || mediaElement.naturalWidth || mediaElement.clientWidth || 0,
    height: mediaElement.videoHeight || mediaElement.naturalHeight || mediaElement.clientHeight || 0
  };
}

function getDisplaySize(mediaElement) {
  const rect = mediaElement.getBoundingClientRect();
  return {
    width: Math.max(1, Math.round(rect.width || mediaElement.clientWidth || 0)),
    height: Math.max(1, Math.round(rect.height || mediaElement.clientHeight || 0))
  };
}

export function getFeatureNames() {
  const names = [];
  for (const kp of KEYPOINT_NAMES) {
    names.push(`${kp}_x`);
    names.push(`${kp}_y`);
  }
  return names;
}

export function poseToFeatures(pose) {
  if (!pose?.keypoints?.length) {
    return null;
  }

  const keypoints = pose.keypoints;
  const vis = keypoints.filter((p) => (p.score ?? 0) >= FEATURE_MIN_SCORE);
  if (vis.length < MIN_VISIBLE_FEATURE_POINTS) {
    return null;
  }

  const xs = vis.map((p) => p.x);
  const ys = vis.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const width = Math.max(maxX - minX, 1);
  const height = Math.max(maxY - minY, 1);

  const byName = Object.create(null);
  for (const p of keypoints) {
    if (p?.name) {
      byName[p.name] = p;
    }
  }

  const features = [];
  for (const name of KEYPOINT_NAMES) {
    const p = byName[name];
    if (!p || (p.score ?? 0) < FEATURE_MIN_SCORE) {
      features.push(0, 0);
    } else {
      features.push((p.x - cx) / width, (p.y - cy) / height);
    }
  }

  return {
    features,
    keypoints,
    bbox: { minX, maxX, minY, maxY, cx, cy, width, height }
  };
}

export function drawPose(canvas, mediaElement, pose) {
  if (!canvas || !mediaElement) {
    return;
  }

  const source = getSourceSize(mediaElement);
  const display = getDisplaySize(mediaElement);
  if (!source.width || !source.height || !display.width || !display.height) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${display.width}px`;
  canvas.style.height = `${display.height}px`;
  canvas.width = Math.round(display.width * dpr);
  canvas.height = Math.round(display.height * dpr);

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, display.width, display.height);

  if (!pose?.keypoints?.length) {
    return;
  }

  const scaleX = display.width / source.width;
  const scaleY = display.height / source.height;

  const pairs = [
    ['nose', 'leftEye'],
    ['nose', 'rightEye'],
    ['leftEye', 'leftEar'],
    ['rightEye', 'rightEar'],
    ['nose', 'leftShoulder'],
    ['nose', 'rightShoulder'],
    ['leftShoulder', 'rightShoulder'],
    ['leftShoulder', 'leftElbow'],
    ['leftElbow', 'leftWrist'],
    ['rightShoulder', 'rightElbow'],
    ['rightElbow', 'rightWrist'],
    ['leftShoulder', 'leftHip'],
    ['rightShoulder', 'rightHip'],
    ['leftHip', 'rightHip'],
    ['leftHip', 'leftKnee'],
    ['leftKnee', 'leftAnkle'],
    ['rightHip', 'rightKnee'],
    ['rightKnee', 'rightAnkle']
  ];

  const byName = Object.create(null);
  for (const p of pose.keypoints) {
    if (p?.name) {
      byName[p.name] = p;
    }
  }

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#1eff63';
  for (const [a, b] of pairs) {
    const pa = byName[a];
    const pb = byName[b];
    if (!pa || !pb || (pa.score ?? 0) < DRAW_MIN_SCORE || (pb.score ?? 0) < DRAW_MIN_SCORE) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(pa.x * scaleX, pa.y * scaleY);
    ctx.lineTo(pb.x * scaleX, pb.y * scaleY);
    ctx.stroke();
  }

  const ls = byName.leftShoulder;
  const rs = byName.rightShoulder;
  const lh = byName.leftHip;
  const rh = byName.rightHip;
  if (ls && rs && lh && rh) {
    const shoulderMid = {
      x: (ls.x + rs.x) / 2,
      y: (ls.y + rs.y) / 2,
      score: Math.min(ls.score ?? 0, rs.score ?? 0)
    };
    const hipMid = {
      x: (lh.x + rh.x) / 2,
      y: (lh.y + rh.y) / 2,
      score: Math.min(lh.score ?? 0, rh.score ?? 0)
    };
    if (shoulderMid.score >= DRAW_MIN_SCORE && hipMid.score >= DRAW_MIN_SCORE) {
      ctx.beginPath();
      ctx.moveTo(shoulderMid.x * scaleX, shoulderMid.y * scaleY);
      ctx.lineTo(hipMid.x * scaleX, hipMid.y * scaleY);
      ctx.stroke();
    }
  }

  const visible = pose.keypoints.filter((p) => (p.score ?? 0) >= DRAW_MIN_SCORE);
  if (visible.length >= 6) {
    const xs = visible.map((p) => p.x * scaleX);
    const ys = visible.map((p) => p.y * scaleY);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys);
    ctx.strokeStyle = '#3cff57';
    ctx.lineWidth = 2;
    ctx.strokeRect(minX, minY, Math.max(1, maxX - minX), Math.max(1, maxY - minY));
  }

  for (const p of pose.keypoints) {
    if ((p.score ?? 0) < DRAW_MIN_SCORE) {
      continue;
    }
    ctx.fillStyle = '#2cb6ff';
    ctx.beginPath();
    ctx.arc(p.x * scaleX, p.y * scaleY, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}
