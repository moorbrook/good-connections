import assert from 'node:assert/strict'
import {
  DEFAULT_GUARD_THRESHOLDS,
  boardTransitionPass,
  diagnosticLogit,
  diagnosticsPass,
  meanAbsolutePixelDelta,
  repeatedObservationPass,
} from '../extension/guard.js'

const thresholds = {
  ...DEFAULT_GUARD_THRESHOLDS,
  diagnosticBias: 0,
  diagnosticWeights: [0, 1, 0, 0],
  minDiagnosticLogit: 0.5,
  maxRepeatPixelDelta: 0.01,
  minBoardPixelDelta: 0.1,
}
const observation = {
  choice: [1, 4, 8],
  cardCount: 12,
  pixels: [0.1, 0.2, 0.3, 0.4],
  score: 2,
  scoreMargin: 0.8,
  attributeEntropy: 0.1,
  attributeMargin: 0.9,
}

assert.equal(meanAbsolutePixelDelta(observation.pixels, [...observation.pixels]), 0)
assert.equal(diagnosticLogit(observation, thresholds), 0.8)
assert.equal(diagnosticsPass(observation, thresholds), true)
assert.equal(repeatedObservationPass(observation, { ...observation, pixels: [0.101, 0.199, 0.302, 0.398] }, thresholds), true)
assert.equal(repeatedObservationPass(observation, { ...observation, choice: [1, 5, 8] }, thresholds), false)
assert.equal(repeatedObservationPass(observation, { ...observation, scoreMargin: 0.2 }, thresholds), false)

const changed = { ...observation, pixels: [0.8, 0.7, 0.6, 0.5] }
const changedRepeat = { ...changed, pixels: [0.801, 0.699, 0.602, 0.498] }
assert.equal(boardTransitionPass(observation, changed, changedRepeat, thresholds), true)
assert.equal(boardTransitionPass(observation, observation, observation, thresholds), false)
assert.equal(boardTransitionPass(observation, changed, observation, thresholds), false)

console.log('vision abstention and closed-loop guards: pass')
