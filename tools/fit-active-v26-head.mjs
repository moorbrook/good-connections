#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import {
  BASE_FEATURES,
  MAX_ITERATIONS,
  QUANTILES,
  TOLERANCE,
  TOP_SCORE_COUNT,
  deployedAction,
  evaluate,
  fitLogistic,
  loadRecords,
  standardize,
} from './fit-active-v19-head.mjs'

const L2 = 0.01

export function thresholdBoundaries(logits) {
  if (!Array.isArray(logits) || logits.length === 0 || logits.some((value) => !Number.isFinite(value))) {
    throw new Error('threshold logits must be a non-empty finite array')
  }
  const ordered = [...new Set(logits)].sort((left, right) => left - right)
  const lowerPadding = Math.max(1, Math.abs(ordered[0]))
  const upperPadding = Math.max(1, Math.abs(ordered.at(-1)))
  const thresholds = [ordered[0] - lowerPadding]
  for (let index = 1; index < ordered.length; index++) {
    thresholds.push((ordered[index - 1] + ordered[index]) / 2)
  }
  thresholds.push(ordered.at(-1) + upperPadding)
  return thresholds
}

export function selectConstrainedCandidate(candidates, limits) {
  const eligible = candidates.filter((candidate) =>
    candidate.metrics.hasSetFailures <= limits.hasSetFailures
    && candidate.metrics.noSetFailures <= limits.noSetFailures)
  eligible.sort((left, right) =>
    left.metrics.failures - right.metrics.failures
    || right.metrics.worstConditional - left.metrics.worstConditional
    || left.threshold - right.threshold)
  if (eligible.length === 0) throw new Error('no threshold satisfies the conditional constraints')
  return eligible[0]
}

function logitFor(record, transform, fitted) {
  return transform(record).reduce(
    (total, value, index) => total + value * fitted.weights[index],
    fitted.bias,
  )
}

function thresholdCandidates(records, transform, fitted) {
  const logits = records.map((record) => logitFor(record, transform, fitted))
  const logitByRecord = new Map(records.map((record, index) => [record, logits[index]]))
  return thresholdBoundaries(logits).map((threshold) => ({
    threshold,
    metrics: evaluate(records, (record) => logitByRecord.get(record) >= threshold),
  }))
}

export async function main() {
  const [evidenceRootArg, deployedHeadArg, outputHeadArg, reportArg, extra] = process.argv.slice(2)
  if (!evidenceRootArg || !deployedHeadArg || !outputHeadArg || !reportArg || extra) {
    throw new Error('usage: fit-active-v26-head EVIDENCE_ROOT DEPLOYED_HEAD OUTPUT_HEAD REPORT')
  }

  const records = await loadRecords(resolve(evidenceRootArg))
  const fitting = records.slice(0, 8192)
  const validation = records.slice(8192)
  const deployedHead = JSON.parse(await readFile(resolve(deployedHeadArg), 'utf8'))
  const deployedFitting = evaluate(fitting, (record) => deployedAction(record, deployedHead))
  const deployedValidation = evaluate(validation, (record) => deployedAction(record, deployedHead))

  const { means, stddevs, transform } = standardize(fitting)
  const fitted = fitLogistic(fitting, transform, L2)
  const selected = selectConstrainedCandidate(
    thresholdCandidates(fitting, transform, fitted),
    {
      hasSetFailures: deployedFitting.hasSetFailures,
      noSetFailures: deployedFitting.noSetFailures,
    },
  )
  const validationMetrics = evaluate(
    validation,
    (record) => logitFor(record, transform, fitted) >= selected.threshold,
  )
  const validationTolerance = 4
  const eligible = validationMetrics.failures < deployedValidation.failures
    && validationMetrics.hasSetFailures <= deployedValidation.hasSetFailures + validationTolerance
    && validationMetrics.noSetFailures <= deployedValidation.noSetFailures + validationTolerance

  const config = {
    type: 'logistic-score-distribution',
    baseFeatures: BASE_FEATURES,
    tripleScoreCount: 220,
    topScoreCount: TOP_SCORE_COUNT,
    quantiles: QUANTILES,
    means,
    stddevs,
    weights: fitted.weights,
    bias: fitted.bias - selected.threshold,
  }
  const report = {
    protocol: {
      fitting: [0, 8192],
      validation: [8192, 16384],
      l2: L2,
      solver: { type: 'IRLS', maxIterations: MAX_ITERATIONS, tolerance: TOLERANCE },
      thresholdConstraint: 'candidate fitting failures no worse than deployed in either board kind',
      validationTolerancePerKind: validationTolerance,
    },
    deployed: { fitting: deployedFitting, validation: deployedValidation },
    candidate: {
      fitting: selected.metrics,
      validation: validationMetrics,
      originalBias: fitted.bias,
      selectedThreshold: selected.threshold,
      adjustedBias: config.bias,
      iterations: fitted.iterations,
    },
    eligible,
  }
  await writeFile(resolve(outputHeadArg), `${JSON.stringify(config, null, 2)}\n`, { flag: 'wx' })
  await writeFile(resolve(reportArg), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify(report))
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack ?? String(error))
    process.exitCode = 1
  })
}
