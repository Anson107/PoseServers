function mean(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((acc, cur) => acc + cur, 0) / values.length;
}

function mse(values) {
  if (!values.length) {
    return 0;
  }
  const m = mean(values);
  return values.reduce((acc, cur) => acc + (cur - m) ** 2, 0) / values.length;
}

function pickRandom(list, count) {
  const copy = [...list];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, Math.max(1, count));
}

function bootstrapIndices(size) {
  const out = [];
  for (let i = 0; i < size; i += 1) {
    out.push(Math.floor(Math.random() * size));
  }
  return out;
}

function bestSplit(samples, targets, indices, featureCandidates) {
  let best = null;
  const parentValues = indices.map((idx) => targets[idx]);
  const parentMse = mse(parentValues);

  for (const feature of featureCandidates) {
    const featureValues = indices.map((idx) => samples[idx][feature]);
    const min = Math.min(...featureValues);
    const max = Math.max(...featureValues);
    if (min === max) {
      continue;
    }

    for (let i = 1; i <= 8; i += 1) {
      const threshold = min + ((max - min) * i) / 9;
      const left = [];
      const right = [];
      for (const idx of indices) {
        if (samples[idx][feature] <= threshold) {
          left.push(idx);
        } else {
          right.push(idx);
        }
      }
      if (!left.length || !right.length) {
        continue;
      }
      const leftMse = mse(left.map((idx) => targets[idx]));
      const rightMse = mse(right.map((idx) => targets[idx]));
      const weighted = (leftMse * left.length + rightMse * right.length) / indices.length;
      const gain = parentMse - weighted;

      if (!best || gain > best.gain) {
        best = { feature, threshold, gain, left, right };
      }
    }
  }

  return best;
}

function buildTree(samples, targets, featureCount, params, indices, depth, importance) {
  const values = indices.map((idx) => targets[idx]);
  const value = mean(values);

  if (depth >= params.maxDepth || indices.length <= params.minLeaf || mse(values) < 1e-6) {
    return { type: 'leaf', value, count: indices.length, depth };
  }

  const allFeatures = Array.from({ length: featureCount }, (_, i) => i);
  const candidateCount = Math.max(2, Math.floor(Math.sqrt(featureCount)));
  const candidates = pickRandom(allFeatures, candidateCount);
  const split = bestSplit(samples, targets, indices, candidates);

  if (!split || split.gain <= 1e-6) {
    return { type: 'leaf', value, count: indices.length, depth };
  }

  importance[split.feature] = (importance[split.feature] || 0) + split.gain;

  return {
    type: 'node',
    feature: split.feature,
    threshold: split.threshold,
    gain: split.gain,
    value,
    depth,
    left: buildTree(samples, targets, featureCount, params, split.left, depth + 1, importance),
    right: buildTree(samples, targets, featureCount, params, split.right, depth + 1, importance)
  };
}

function predictTree(tree, sample) {
  if (!tree || tree.type === 'leaf') {
    return tree?.value ?? 0;
  }
  if (sample[tree.feature] <= tree.threshold) {
    return predictTree(tree.left, sample);
  }
  return predictTree(tree.right, sample);
}

function treeDepth(tree) {
  if (!tree) {
    return 0;
  }
  if (tree.type === 'leaf') {
    return 1;
  }
  return 1 + Math.max(treeDepth(tree.left), treeDepth(tree.right));
}

export function trainRandomForest(samples, targets, params = {}) {
  const options = {
    trees: Math.max(10, Math.min(500, params.trees ?? 100)),
    maxDepth: Math.max(1, Math.min(12, params.maxDepth ?? 8)),
    minLeaf: Math.max(1, Math.min(20, params.minLeaf ?? 2))
  };

  if (!samples.length) {
    throw new Error('no samples');
  }

  const featureCount = samples[0].length;
  const trees = [];
  const importance = new Array(featureCount).fill(0);

  for (let i = 0; i < options.trees; i += 1) {
    const indices = bootstrapIndices(samples.length);
    const tree = buildTree(samples, targets, featureCount, options, indices, 0, importance);
    trees.push(tree);
  }

  const predictions = samples.map((sample) => predictForest({ trees }, sample));
  const errors = predictions.map((pred, i) => Math.abs(pred - targets[i]));
  const mae = mean(errors);

  const treeMetrics = trees.map((tree, id) => {
    const perTreePred = samples.map((sample) => predictTree(tree, sample));
    const perTreeErrors = perTreePred.map((pred, i) => Math.abs(pred - targets[i]));
    return { id, mae: mean(perTreeErrors), depth: treeDepth(tree) };
  });

  return {
    version: 1,
    algorithm: 'simple-rf-regression',
    params: options,
    trees,
    importance,
    metrics: {
      mae,
      minMae: Math.min(...treeMetrics.map((t) => t.mae)),
      maxMae: Math.max(...treeMetrics.map((t) => t.mae)),
      avgDepth: mean(treeMetrics.map((t) => t.depth)),
      maxDepth: Math.max(...treeMetrics.map((t) => t.depth))
    },
    treeMetrics
  };
}

export function predictForest(model, sample) {
  if (!model?.trees?.length) {
    return 0;
  }
  const values = model.trees.map((tree) => predictTree(tree, sample));
  return mean(values);
}

export function normalizeImportance(importance, names) {
  const total = importance.reduce((acc, cur) => acc + cur, 0) || 1;
  return importance
    .map((v, i) => ({ name: names[i] || `f${i}`, score: v / total }))
    .sort((a, b) => b.score - a.score);
}

function scoreByBaseRules(featureVector) {
  // Shoulders and elbows extend ratio can reflect draw posture in archery.
  const lsx = featureVector[10] || 0;
  const rsx = featureVector[12] || 0;
  const lex = featureVector[14] || 0;
  const rex = featureVector[16] || 0;
  const wristSpread = Math.abs((featureVector[18] || 0) - (featureVector[20] || 0));
  const shoulderWidth = Math.abs(rsx - lsx);
  const elbowSpread = Math.abs(rex - lex);
  const stability = Math.max(0, 1 - Math.abs(shoulderWidth - elbowSpread));

  const score = (0.5 * stability + 0.3 * wristSpread + 0.2 * shoulderWidth) * 100;
  return Math.max(0, Math.min(100, score));
}

export function scoreByRules(featureVector, referenceFeatureVector = null) {
  if (!featureVector?.length) {
    return 0;
  }

  if (referenceFeatureVector?.length === featureVector.length) {
    let sumDist = 0;
    let matched = 0;

    // Features are [x0,y0,x1,y1,...]. Compare only keypoints present in both vectors.
    for (let i = 0; i < featureVector.length; i += 2) {
      const ax = featureVector[i] ?? 0;
      const ay = featureVector[i + 1] ?? 0;
      const bx = referenceFeatureVector[i] ?? 0;
      const by = referenceFeatureVector[i + 1] ?? 0;
      const hasA = !(ax === 0 && ay === 0);
      const hasB = !(bx === 0 && by === 0);

      if (!hasA || !hasB) {
        continue;
      }

      const dx = ax - bx;
      const dy = ay - by;
      sumDist += Math.sqrt(dx * dx + dy * dy);
      matched += 1;
    }

    // Too few shared joints: degrade gracefully to base rule score.
    if (matched < 4) {
      return scoreByBaseRules(featureVector);
    }

    const meanDist = sumDist / matched;
    const similarity = Math.max(0, 1 - Math.min(1, meanDist / 0.65));
    const coverage = Math.max(0, Math.min(1, matched / 17));
    const score = (0.8 * similarity + 0.2 * coverage) * 100;
    return Math.max(0, Math.min(100, score));
  }

  return scoreByBaseRules(featureVector);
}
