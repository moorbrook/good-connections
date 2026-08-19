import { readFile } from 'node:fs/promises'

const [samplesPath, minAcceptedAccuracyText = '1'] = process.argv.slice(2)
if (!samplesPath) {
  console.error('Usage: node tools/calibrate-vision-guard.mjs ABSTENTION_SAMPLES.tsv [MIN_ACCEPTED_ACCURACY]')
  process.exit(2)
}
const minAcceptedAccuracy = Number(minAcceptedAccuracyText)
if (!(minAcceptedAccuracy > 0 && minAcceptedAccuracy <= 1)) {
  throw new Error('MIN_ACCEPTED_ACCURACY must be greater than zero and at most one')
}

const rows = (await readFile(samplesPath, 'utf8')).trim().split('\n')
const header = rows.shift()?.split('\t') ?? []
const column = Object.fromEntries(header.map((name, index) => [name, index]))
const required = ['correct', 'strongest_is_set', 'score', 'score_margin', 'mean_attribute_entropy', 'min_attribute_margin']
for (const name of required) {
  if (!(name in column)) throw new Error(`Missing ${name} column in ${samplesPath}`)
}

const actionRows = rows.filter((row) => row.split('\t')[column.strongest_is_set] === '1')
const samples = actionRows.map((row) => {
  const values = row.split('\t')
  return {
    label: Number(values[column.correct]),
    features: [
      Number(values[column.score]),
      Number(values[column.score_margin]),
      -Number(values[column.mean_attribute_entropy]),
      Number(values[column.min_attribute_margin]),
    ],
  }
})
if (samples.length === 0 || samples.some((sample) => sample.features.some((value) => !Number.isFinite(value)))) {
  throw new Error('Samples must contain finite diagnostics')
}

const featureCount = samples[0].features.length
const means = Array.from({ length: featureCount }, (_, feature) =>
  samples.reduce((total, sample) => total + sample.features[feature], 0) / samples.length)
const deviations = Array.from({ length: featureCount }, (_, feature) => {
  const variance = samples.reduce((total, sample) => total + (sample.features[feature] - means[feature]) ** 2, 0) / samples.length
  return Math.max(Math.sqrt(variance), 1e-9)
})
const normalized = samples.map((sample) => ({
  label: sample.label,
  features: sample.features.map((value, feature) => (value - means[feature]) / deviations[feature]),
}))
const positives = normalized.filter((sample) => sample.label === 1).length
const negatives = normalized.length - positives
if (positives === 0 || negatives === 0) {
  throw new Error(`Guard fitting requires both correct and incorrect research boards; got ${positives} correct and ${negatives} incorrect`)
}

const sigmoid = (value) => value >= 0
  ? 1 / (1 + Math.exp(-value))
  : Math.exp(value) / (1 + Math.exp(value))
let bias = 0
const weights = new Array(featureCount).fill(0)
const positiveWeight = normalized.length / (2 * positives)
const negativeWeight = normalized.length / (2 * negatives)
for (let step = 0; step < 6_000; step++) {
  let biasGradient = 0
  const gradients = new Array(featureCount).fill(0)
  for (const sample of normalized) {
    const logit = bias + sample.features.reduce((total, value, feature) => total + value * weights[feature], 0)
    const sampleWeight = sample.label === 1 ? positiveWeight : negativeWeight
    const error = (sigmoid(logit) - sample.label) * sampleWeight
    biasGradient += error
    for (let feature = 0; feature < featureCount; feature++) gradients[feature] += error * sample.features[feature]
  }
  const learningRate = 0.08 / (1 + step / 1_500)
  bias -= learningRate * biasGradient / normalized.length
  for (let feature = 0; feature < featureCount; feature++) {
    weights[feature] -= learningRate * (gradients[feature] / normalized.length + 1e-3 * weights[feature])
  }
}

const rawWeights = weights.map((weight, feature) => weight / deviations[feature])
const rawBias = bias - rawWeights.reduce((total, weight, feature) => total + weight * means[feature], 0)
const logits = samples.map((sample) => rawBias + sample.features.reduce((total, value, feature) => total + value * rawWeights[feature], 0))
const ranked = logits
  .map((logit, index) => ({ logit, label: samples[index].label }))
  .sort((left, right) => right.logit - left.logit)
let thresholdLogit = Number.POSITIVE_INFINITY
let rankedAccepted = 0
let rankedCorrect = 0
for (let start = 0; start < ranked.length;) {
  let end = start
  while (end < ranked.length && ranked[end].logit === ranked[start].logit) {
    rankedCorrect += ranked[end].label
    end++
  }
  rankedAccepted = end
  if (rankedCorrect / rankedAccepted >= minAcceptedAccuracy) {
    thresholdLogit = ranked[start].logit
  }
  start = end
}
if (!Number.isFinite(thresholdLogit)) {
  throw new Error(`No operating point reaches ${minAcceptedAccuracy} accepted accuracy`)
}
const accepted = logits.map((logit) => logit >= thresholdLogit)
const acceptedCorrect = accepted.filter((value, index) => value && samples[index].label === 1).length
const acceptedIncorrect = accepted.filter((value, index) => value && samples[index].label === 0).length

console.log(JSON.stringify({
  featureOrder: ['score', 'scoreMargin', 'negativeAttributeEntropy', 'attributeMargin'],
  bias: rawBias,
  weights: rawWeights,
  minAcceptedAccuracy,
  thresholdLogit,
  excludedNoSetActions: rows.length - actionRows.length,
  samples: samples.length,
  correct: positives,
  incorrect: negatives,
  accepted: acceptedCorrect + acceptedIncorrect,
  acceptedCorrect,
  acceptedIncorrect,
  coverage: (acceptedCorrect + acceptedIncorrect) / samples.length,
}, null, 2))
