#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

export const BASE_FEATURES = [
  'score',
  'scoreMargin',
  'attributeEntropy',
  'attributeMargin',
  'scoreMean',
  'scoreStddev',
  'scoreLogsumexp',
  'positiveScoreFraction',
]
export const TOP_SCORE_COUNT = 16
export const QUANTILES = [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1]
export const L2_VALUES = [0.01, 0.1, 1, 10]
export const MAX_ITERATIONS = 25
export const TOLERANCE = 1e-9

export function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value))
  const exponential = Math.exp(value)
  return exponential / (1 + exponential)
}

export function cardAttributes(id) {
  return [Math.floor(id / 27) % 3, Math.floor(id / 9) % 3, Math.floor(id / 3) % 3, id % 3]
}

export function isSet(ids) {
  const cards = ids.map(cardAttributes)
  return [0, 1, 2, 3].every((attribute) =>
    (cards[0][attribute] + cards[1][attribute] + cards[2][attribute]) % 3 === 0)
}

export function richerFeatures(record) {
  const descending = record.sortedTripleScores
  if (!Array.isArray(descending) || descending.length !== 220 || descending.some((value) => !Number.isFinite(value))) {
    throw new Error(`record ${record.index} has invalid sorted triple scores`)
  }
  const ascending = [...descending].reverse()
  return [
    ...BASE_FEATURES.map((feature) => record[feature]),
    ...descending.slice(0, TOP_SCORE_COUNT),
    ...QUANTILES.map((quantile) => ascending[Math.round(quantile * (ascending.length - 1))]),
  ]
}

export function standardize(records) {
  const vectors = records.map(richerFeatures)
  const width = vectors[0].length
  const means = Array.from({ length: width }, (_, column) =>
    vectors.reduce((total, row) => total + row[column], 0) / vectors.length)
  const stddevs = Array.from({ length: width }, (_, column) => {
    const variance = vectors.reduce((total, row) => total + (row[column] - means[column]) ** 2, 0) / vectors.length
    const value = Math.sqrt(variance)
    if (!Number.isFinite(value) || value <= 0) throw new Error(`feature ${column} has zero variance`)
    return value
  })
  const transform = (record) => richerFeatures(record).map((value, index) => (value - means[index]) / stddevs[index])
  return { means, stddevs, transform }
}

export function solveLinear(matrix, values) {
  const size = values.length
  const rows = matrix.map((row, index) => [...row, values[index]])
  for (let pivot = 0; pivot < size; pivot++) {
    let best = pivot
    for (let row = pivot + 1; row < size; row++) {
      if (Math.abs(rows[row][pivot]) > Math.abs(rows[best][pivot])) best = row
    }
    if (Math.abs(rows[best][pivot]) < 1e-12) throw new Error('IRLS system is singular')
    ;[rows[pivot], rows[best]] = [rows[best], rows[pivot]]
    const divisor = rows[pivot][pivot]
    for (let column = pivot; column <= size; column++) rows[pivot][column] /= divisor
    for (let row = 0; row < size; row++) {
      if (row === pivot) continue
      const factor = rows[row][pivot]
      for (let column = pivot; column <= size; column++) rows[row][column] -= factor * rows[pivot][column]
    }
  }
  return rows.map((row) => row[size])
}

export function fitLogistic(records, transform, l2) {
  const inputs = records.map((record) => [1, ...transform(record)])
  const labels = records.map((record) => Number(record.kind === 'has-set'))
  const width = inputs[0].length
  let coefficients = Array(width).fill(0)
  let iterations = 0
  for (; iterations < MAX_ITERATIONS; iterations++) {
    const system = Array.from({ length: width }, () => Array(width).fill(0))
    const target = Array(width).fill(0)
    for (let row = 0; row < inputs.length; row++) {
      const input = inputs[row]
      const logit = input.reduce((total, value, index) => total + value * coefficients[index], 0)
      const probability = sigmoid(logit)
      const weight = Math.max(1e-6, probability * (1 - probability))
      const working = logit + (labels[row] - probability) / weight
      for (let left = 0; left < width; left++) {
        target[left] += weight * input[left] * working
        for (let right = 0; right < width; right++) system[left][right] += weight * input[left] * input[right]
      }
    }
    for (let diagonal = 1; diagonal < width; diagonal++) system[diagonal][diagonal] += l2 * inputs.length
    const next = solveLinear(system, target)
    const delta = Math.max(...next.map((value, index) => Math.abs(value - coefficients[index])))
    coefficients = next
    if (delta < TOLERANCE) break
  }
  return { bias: coefficients[0], weights: coefficients.slice(1), iterations: iterations + 1 }
}

export function deployedAction(record, head) {
  const logit = BASE_FEATURES.reduce((total, feature, index) =>
    total + head.weights[index] * (record[feature] - head.means[index]) / head.stddevs[index], head.bias)
  return logit >= 0
}

export function actionCorrect(record, predictsSet) {
  if (!predictsSet) return record.kind === 'no-set'
  return isSet(record.strongestPositions.map((position) => record.boardIds[position]))
}

export function evaluate(records, action) {
  const result = {
    boards: records.length,
    failures: 0,
    hasSetBoards: 0,
    hasSetFailures: 0,
    noSetBoards: 0,
    noSetFailures: 0,
  }
  for (const record of records) {
    const prefix = record.kind === 'has-set' ? 'hasSet' : 'noSet'
    result[`${prefix}Boards`]++
    if (!actionCorrect(record, action(record))) {
      result.failures++
      result[`${prefix}Failures`]++
    }
  }
  result.accuracy = 1 - result.failures / result.boards
  result.hasSetAccuracy = 1 - result.hasSetFailures / result.hasSetBoards
  result.noSetAccuracy = 1 - result.noSetFailures / result.noSetBoards
  result.worstConditional = Math.min(result.hasSetAccuracy, result.noSetAccuracy)
  return result
}

export function candidateRank(left, right) {
  return left.validation.failures - right.validation.failures
    || right.validation.worstConditional - left.validation.worstConditional
    || left.featureCount - right.featureCount
    || (right.l2 ?? Number.POSITIVE_INFINITY) - (left.l2 ?? Number.POSITIVE_INFINITY)
}

export async function loadRecords(root) {
  const records = []
  for (const offset of [0, 4096, 8192, 12288]) {
    const text = await readFile(resolve(root, `shard-${offset}`, 'evidence.jsonl'), 'utf8')
    records.push(...text.split('\n').filter(Boolean).map((line) => JSON.parse(line)))
  }
  if (records.length !== 16_384) throw new Error(`expected 16384 evidence rows, got ${records.length}`)
  records.forEach((record, index) => {
    if (record.index !== index) throw new Error(`evidence index ${record.index} at row ${index}`)
  })
  return records
}

export async function main() {
  const [rootArg, deployedHeadArg, outputHeadArg, reportArg, extra] = process.argv.slice(2)
  if (!rootArg || !deployedHeadArg || !outputHeadArg || !reportArg || extra) {
    throw new Error('usage: fit-active-v19-head EVIDENCE_ROOT DEPLOYED_HEAD OUTPUT_HEAD REPORT')
  }
  const records = await loadRecords(resolve(rootArg))
  const fitting = records.slice(0, 8192)
  const validation = records.slice(8192)
  const deployedHead = JSON.parse(await readFile(resolve(deployedHeadArg), 'utf8'))
  const deployed = {
    kind: 'deployed',
    featureCount: BASE_FEATURES.length,
    fitting: evaluate(fitting, (record) => deployedAction(record, deployedHead)),
    validation: evaluate(validation, (record) => deployedAction(record, deployedHead)),
  }
  const { means, stddevs, transform } = standardize(fitting)
  const rich = L2_VALUES.map((l2) => {
    const fitted = fitLogistic(fitting, transform, l2)
    const action = (record) => transform(record).reduce(
      (total, value, index) => total + value * fitted.weights[index],
      fitted.bias,
    ) >= 0
    return {
      kind: 'rich-logistic',
      featureCount: fitted.weights.length,
      l2,
      ...fitted,
      fitting: evaluate(fitting, action),
      validation: evaluate(validation, action),
    }
  })
  const tolerance = 4 / 4096
  const eligible = rich.filter((candidate) =>
    candidate.validation.hasSetAccuracy + tolerance >= deployed.validation.hasSetAccuracy
    && candidate.validation.noSetAccuracy + tolerance >= deployed.validation.noSetAccuracy)
  const candidates = [deployed, ...eligible].sort(candidateRank)
  const selected = candidates[0]
  const selectedHead = selected.kind === 'deployed'
    ? deployedHead
    : {
        type: 'logistic-score-distribution',
        baseFeatures: BASE_FEATURES,
        topScoreCount: TOP_SCORE_COUNT,
        quantiles: QUANTILES,
        means,
        stddevs,
        weights: selected.weights,
        bias: selected.bias,
      }
  await writeFile(resolve(outputHeadArg), `${JSON.stringify(selectedHead, null, 2)}\n`, { flag: 'wx' })
  await writeFile(resolve(reportArg), `${JSON.stringify({
    protocol: {
      fitting: [0, 8192],
      validation: [8192, 16384],
      baseFeatures: BASE_FEATURES,
      topScoreCount: TOP_SCORE_COUNT,
      quantiles: QUANTILES,
      l2Values: L2_VALUES,
      solver: { type: 'IRLS', maxIterations: MAX_ITERATIONS, tolerance: TOLERANCE },
    },
    deployed,
    rich,
    eligibleL2: eligible.map((candidate) => candidate.l2),
    selected: { ...selected, weights: undefined },
  }, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify({
    deployed: deployed.validation,
    rich: rich.map(({ l2, validation, iterations }) => ({ l2, validation, iterations })),
    eligibleL2: eligible.map((candidate) => candidate.l2),
    selected: selected.kind === 'deployed' ? 'deployed' : `rich-l2-${selected.l2}`,
  }))
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack ?? String(error))
    process.exitCode = 1
  })
}
