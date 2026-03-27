export function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

export function toPercent(v, digits = 2) {
  return `${(v * 100).toFixed(digits)}%`;
}
