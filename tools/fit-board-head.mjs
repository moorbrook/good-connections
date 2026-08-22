#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const FEATURES = [
  'score',
  'scoreMargin',
  'attributeEntropy',
  'attributeMargin',
  'scoreMean',
  'scoreStddev',
  'scoreLogsumexp',
  'positiveScoreFraction',
]
const L2_VALUES = [0, 0.001, 0.01, 0.1]
const STEPS = 5_000
const LEARNING_RATE = 0.05

function usage() {
  return 'usage: node tools/fit-board-head.mjs EVIDENCE_JSONL OUTPUT_JSON REPORT_JSON'
}

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value))
  const exponential = Math.exp(value)
  return exponential / (1 + exponential)
}

function mix32(value) {
  value = Math.imul(value ^ (value >>> 16), 0x21f0aaad)
  value = Math.imul(value ^ (value >>> 15), 0x735a2d97)
  return (value ^ (value >>> 15)) >>> 0
}

function stratifiedSplit(records) {
  const fitting = []
  const validation = []
  for (const kind of ['has-set', 'no-set']) {
    const stratum = records
      .filter((record) => record.kind === kind)
      .sort((left, right) => {
        const leftRank = mix32(left.boardSeed ^ 0x4d595df4)
        const rightRank = mix32(right.boardSeed ^ 0x4d595df4)
        return leftRank - rightRank || left.boardSeed - right.boardSeed || left.iteration - right.iteration
      })
    if (stratum.length !== 512) throw new Error(`expected 512 ${kind} records, got ${stratum.length}`)
    fitting.push(...stratum.slice(0, 256))
    validation.push(...stratum.slice(256))
  }
  return { fitting, validation }
}

function featureVector(record) {
  return FEATURES.map((feature) => {
    const value = record[feature]
    if (!Number.isFinite(value)) throw new Error(`record ${record.iteration} has invalid ${feature}`)
    return value
  })
}

function standardization(records) {
  const vectors = records.map(featureVector)
  const means = FEATURES.map((_, feature) => vectors.reduce((sum, vector) => sum + vector[feature], 0) / vectors.length)
  const stddevs = FEATURES.map((_, feature) => {
    const variance = vectors.reduce((sum, vector) => sum + (vector[feature] - means[feature]) ** 2, 0) / vectors.length
    const stddev = Math.sqrt(variance)
    if (!Number.isFinite(stddev) || stddev <= 0) throw new Error(`feature ${FEATURES[feature]} has zero variance`)
    return stddev
  })
  const transform = (record) => featureVector(record).map((value, index) => (value - means[index]) / stddevs[index])
  return { means, stddevs, transform }
}

function fitLogistic(records, transform, l2) {
  const vectors = records.map(transform)
  const labels = records.map((record) => Number(record.kind === 'has-set'))
  const weights = FEATURES.map(() => 0)
  let bias = 0
  for (let step = 0; step < STEPS; step++) {
    const weightGradients = FEATURES.map(() => 0)
    let biasGradient = 0
    for (let row = 0; row < records.length; row++) {
      const logit = vectors[row].reduce((sum, value, index) => sum + value * weights[index], bias)
      const error = sigmoid(logit) - labels[row]
      biasGradient += error
      for (let feature = 0; feature < FEATURES.length; feature++) {
        weightGradients[feature] += error * vectors[row][feature]
      }
    }
    bias -= LEARNING_RATE * biasGradient / records.length
    for (let feature = 0; feature < FEATURES.length; feature++) {
      const gradient = weightGradients[feature] / records.length + l2 * weights[feature]
      weights[feature] -= LEARNING_RATE * gradient
    }
  }
  return { weights, bias }
}

function evaluate(records, actionIsSet) {
  const counts = {
    boards: records.length,
    failures: 0,
    hasSetBoards: 0,
    hasSetFailures: 0,
    noSetBoards: 0,
    noSetFailures: 0,
  }
  for (const record of records) {
    const predictsSet = actionIsSet(record)
    const correct = predictsSet ? record.strongestIsSet : record.kind === 'no-set'
    const prefix = record.kind === 'has-set' ? 'hasSet' : 'noSet'
    counts[`${prefix}Boards`]++
    if (!correct) {
      counts.failures++
      counts[`${prefix}Failures`]++
    }
  }
  counts.accuracy = 1 - counts.failures / counts.boards
  counts.hasSetAccuracy = 1 - counts.hasSetFailures / counts.hasSetBoards
  counts.noSetAccuracy = 1 - counts.noSetFailures / counts.noSetBoards
  counts.worstConditional = Math.min(counts.hasSetAccuracy, counts.noSetAccuracy)
  return counts
}

function candidateRank(left, right) {
  return left.validation.failures - right.validation.failures
    || right.validation.worstConditional - left.validation.worstConditional
    || Number(left.kind !== 'threshold') - Number(right.kind !== 'threshold')
    || (right.l2 ?? 0) - (left.l2 ?? 0)
}

async function main() {
  const [evidenceArg, outputArg, reportArg, extra] = process.argv.slice(2)
  if (!evidenceArg || !outputArg || !reportArg || extra) throw new Error(usage())
  const records = (await readFile(resolve(evidenceArg), 'utf8'))
    .split('\n').filter(Boolean).map((line) => JSON.parse(line))
  if (records.length !== 1_024) throw new Error(`expected 1024 evidence records, got ${records.length}`)
  const { fitting, validation } = stratifiedSplit(records)
  const { means, stddevs, transform } = standardization(fitting)
  const candidates = []

  const orderedScores = fitting.map((record) => record.score).sort((left, right) => left - right)
  const thresholds = []
  for (let index = 1; index < orderedScores.length; index++) {
    if (orderedScores[index] !== orderedScores[index - 1]) {
      thresholds.push((orderedScores[index] + orderedScores[index - 1]) / 2)
    }
  }
  for (const threshold of thresholds) {
    candidates.push({
      kind: 'threshold',
      threshold,
      fitting: evaluate(fitting, (record) => record.score >= threshold),
      validation: evaluate(validation, (record) => record.score >= threshold),
    })
  }

  for (const l2 of L2_VALUES) {
    const { weights, bias } = fitLogistic(fitting, transform, l2)
    const action = (record) => transform(record).reduce((sum, value, index) => sum + value * weights[index], bias) >= 0
    candidates.push({
      kind: 'logistic', l2, weights, bias, means, stddevs,
      fitting: evaluate(fitting, action),
      validation: evaluate(validation, action),
    })
  }

  candidates.sort(candidateRank)
  const selected = candidates[0]
  const config = selected.kind === 'threshold'
    ? { type: 'score-threshold', threshold: selected.threshold }
    : { type: 'logistic', means, stddevs, weights: selected.weights, bias: selected.bias }
  await writeFile(resolve(outputArg), `${JSON.stringify(config, null, 2)}\n`, { flag: 'wx' })
  await writeFile(resolve(reportArg), `${JSON.stringify({
    protocol: {
      split: 'within each board kind, rank by mix32(boardSeed ^ 0x4d595df4); first 256 fit, last 256 validate',
      features: FEATURES,
      l2Values: L2_VALUES,
      steps: STEPS,
      learningRate: LEARNING_RATE,
    },
    fittingRecords: fitting.length,
    validationRecords: validation.length,
    selected,
    candidates: candidates.filter((candidate) => candidate.kind === 'logistic')
      .concat(candidates.filter((candidate) => candidate.kind === 'threshold').slice(0, 10)),
  }, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify({ selected: config, fitting: selected.fitting, validation: selected.validation }))
}

main().catch((error) => {
  console.error(error?.stack ?? String(error))
  process.exitCode = 1
})
