export const DEFAULT_GUARD_THRESHOLDS = Object.freeze({
  // Linear diagnostic guard fitted once on board-v5 research replicate 5.
  // Feature order: score, score margin, negative entropy, attribute margin.
  diagnosticBias: 5.0201131133169445,
  diagnosticWeights: Object.freeze([
    0.3367313801535807,
    -0.0243837612418493,
    121.55320790903255,
    -5.862390950523434,
  ]),
  minDiagnosticLogit: 1.5354155735768167,
  maxRepeatPixelDelta: 0.008,
  minBoardPixelDelta: 0.025,
})

export function meanAbsolutePixelDelta(left, right) {
  if (!left || !right || left.length !== right.length || left.length === 0) return Number.POSITIVE_INFINITY
  let total = 0
  for (let index = 0; index < left.length; index++) total += Math.abs(left[index] - right[index])
  return total / left.length
}

export function choicesEqual(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index])
}

export function diagnosticLogit(observation, thresholds = DEFAULT_GUARD_THRESHOLDS) {
  const features = [
    observation.score,
    observation.scoreMargin,
    -observation.attributeEntropy,
    observation.attributeMargin,
  ]
  if (features.some((value) => !Number.isFinite(value))) return Number.NEGATIVE_INFINITY
  return thresholds.diagnosticBias
    + features.reduce((total, value, index) => total + value * thresholds.diagnosticWeights[index], 0)
}

export function diagnosticsPass(observation, thresholds = DEFAULT_GUARD_THRESHOLDS) {
  return diagnosticLogit(observation, thresholds) >= thresholds.minDiagnosticLogit
}

export function repeatedObservationPass(first, second, thresholds = DEFAULT_GUARD_THRESHOLDS) {
  return choicesEqual(first.choice, second.choice)
    && first.cardCount === second.cardCount
    && meanAbsolutePixelDelta(first.pixels, second.pixels) <= thresholds.maxRepeatPixelDelta
    && diagnosticsPass(second, thresholds)
}

export function boardTransitionPass(before, firstAfter, secondAfter, thresholds = DEFAULT_GUARD_THRESHOLDS) {
  const afterStable = firstAfter.cardCount === secondAfter.cardCount
    && meanAbsolutePixelDelta(firstAfter.pixels, secondAfter.pixels) <= thresholds.maxRepeatPixelDelta
  const changed = before.cardCount !== secondAfter.cardCount
    || meanAbsolutePixelDelta(before.pixels, secondAfter.pixels) >= thresholds.minBoardPixelDelta
  return afterStable && changed
}
