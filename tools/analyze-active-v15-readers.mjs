#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const ATTRIBUTE_COUNT = 4
const ATTRIBUTE_VALUES = 3
const CARD_COUNT = 12
const EXPECTED_TRIPLES = 220
const EPSILON = 1e-12

function mix32(value) {
  value = Math.imul(value ^ (value >>> 16), 0x21f0aaad)
  value = Math.imul(value ^ (value >>> 15), 0x735a2d97)
  return (value ^ (value >>> 15)) >>> 0
}

function triples() {
  const output = []
  for (let first = 0; first < CARD_COUNT - 2; first++) {
    for (let second = first + 1; second < CARD_COUNT - 1; second++) {
      for (let third = second + 1; third < CARD_COUNT; third++) output.push([first, second, third])
    }
  }
  return output
}

const TRIPLES = triples()

function cardAttributes(id) {
  return [Math.floor(id / 27) % 3, Math.floor(id / 9) % 3, Math.floor(id / 3) % 3, id % 3]
}

function isSet(ids) {
  const cards = ids.map(cardAttributes)
  return Array.from({ length: ATTRIBUTE_COUNT }, (_, attribute) => attribute)
    .every((attribute) => (cards[0][attribute] + cards[1][attribute] + cards[2][attribute]) % 3 === 0)
}

function splitRecords(records) {
  const fitting = []
  const validation = []
  for (const kind of ['has-set', 'no-set']) {
    const stratum = records.filter((record) => record.kind === kind).sort((left, right) => {
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

function validateRecord(record) {
  if (!Array.isArray(record.boardIds) || record.boardIds.length !== CARD_COUNT) {
    throw new Error(`iteration ${record.iteration}: invalid board IDs`)
  }
  if (!Array.isArray(record.attributeProbabilities)
    || record.attributeProbabilities.length !== CARD_COUNT * ATTRIBUTE_COUNT * ATTRIBUTE_VALUES
    || record.attributeProbabilities.some((value) => !Number.isFinite(value))) {
    throw new Error(`iteration ${record.iteration}: invalid attribute probabilities`)
  }
  if (!Array.isArray(record.tripleScores) || record.tripleScores.length !== EXPECTED_TRIPLES
    || record.tripleScores.some((value) => !Number.isFinite(value))) {
    throw new Error(`iteration ${record.iteration}: invalid learned triple scores`)
  }
}

function probability(record, card, attribute, value) {
  return record.attributeProbabilities[(card * ATTRIBUTE_COUNT + attribute) * ATTRIBUTE_VALUES + value]
}

function exactScores(record) {
  return TRIPLES.map((triple) => {
    let score = 0
    for (let attribute = 0; attribute < ATTRIBUTE_COUNT; attribute++) {
      let same = 0
      for (let value = 0; value < ATTRIBUTE_VALUES; value++) {
        same += probability(record, triple[0], attribute, value)
          * probability(record, triple[1], attribute, value)
          * probability(record, triple[2], attribute, value)
      }
      let different = 0
      for (let first = 0; first < ATTRIBUTE_VALUES; first++) {
        for (let second = 0; second < ATTRIBUTE_VALUES; second++) {
          if (second === first) continue
          for (let third = 0; third < ATTRIBUTE_VALUES; third++) {
            if (third === first || third === second) continue
            different += probability(record, triple[0], attribute, first)
              * probability(record, triple[1], attribute, second)
              * probability(record, triple[2], attribute, third)
          }
        }
      }
      score += Math.log(Math.max(same + different, EPSILON))
    }
    return score
  })
}

function selectedTriple(scores) {
  let selected = 0
  for (let index = 1; index < scores.length; index++) {
    if (scores[index] > scores[selected]) selected = index
  }
  return { positions: TRIPLES[selected], score: scores[selected] }
}

function actionCorrect(record, action) {
  if (!action.actionIsSet) return record.kind === 'no-set'
  return record.kind === 'has-set' && isSet(action.positions.map((position) => record.boardIds[position]))
}

function evaluate(records, actionForRecord) {
  const metrics = {
    boards: records.length,
    failures: 0,
    hasSetBoards: 0,
    hasSetFailures: 0,
    noSetBoards: 0,
    noSetFailures: 0,
  }
  for (const record of records) {
    const prefix = record.kind === 'has-set' ? 'hasSet' : 'noSet'
    metrics[`${prefix}Boards`]++
    if (!actionCorrect(record, actionForRecord(record))) {
      metrics.failures++
      metrics[`${prefix}Failures`]++
    }
  }
  metrics.accuracy = 1 - metrics.failures / metrics.boards
  metrics.hasSetAccuracy = 1 - metrics.hasSetFailures / metrics.hasSetBoards
  metrics.noSetAccuracy = 1 - metrics.noSetFailures / metrics.noSetBoards
  metrics.worstConditional = Math.min(metrics.hasSetAccuracy, metrics.noSetAccuracy)
  return metrics
}

function thresholdCandidates(fitting, scoreForRecord) {
  const maxima = fitting.map((record) => selectedTriple(scoreForRecord(record)).score)
    .sort((left, right) => left - right)
  const output = []
  for (let index = 1; index < maxima.length; index++) {
    if (maxima[index] !== maxima[index - 1]) output.push((maxima[index] + maxima[index - 1]) / 2)
  }
  return output
}

function fitThreshold(name, fitting, validation, scoreForRecord, details = {}) {
  const candidates = thresholdCandidates(fitting, scoreForRecord).map((threshold) => {
    const action = (record) => {
      const selected = selectedTriple(scoreForRecord(record))
      return { positions: selected.positions, actionIsSet: selected.score >= threshold }
    }
    return { threshold, fitting: evaluate(fitting, action), validation: evaluate(validation, action) }
  })
  candidates.sort((left, right) => left.fitting.failures - right.fitting.failures
    || right.fitting.worstConditional - left.fitting.worstConditional
    || right.threshold - left.threshold)
  const selected = candidates[0]
  if (!selected) throw new Error(`${name}: no threshold candidates`)
  return { name, ...details, ...selected }
}

function scoreMoments(records, scoreForRecord) {
  const values = records.flatMap(scoreForRecord)
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
  const stddev = Math.sqrt(variance)
  if (!Number.isFinite(stddev) || stddev <= 0) throw new Error('score family has zero variance')
  return { mean, stddev }
}

function discreteAction(record) {
  const attributes = Array.from({ length: CARD_COUNT }, (_, card) =>
    Array.from({ length: ATTRIBUTE_COUNT }, (_, attribute) => {
      const values = Array.from(
        { length: ATTRIBUTE_VALUES },
        (_, value) => probability(record, card, attribute, value),
      )
      return values.indexOf(Math.max(...values))
    }))
  const valid = TRIPLES.map((positions, index) => ({
    positions,
    index,
    valid: Array.from({ length: ATTRIBUTE_COUNT }, (_, attribute) => attribute)
      .every((attribute) => positions.reduce((sum, position) => sum + attributes[position][attribute], 0) % 3 === 0),
  })).filter((candidate) => candidate.valid)
  if (valid.length === 0) return { positions: TRIPLES[0], actionIsSet: false }
  const exact = record.exactScores
  valid.sort((left, right) => exact[right.index] - exact[left.index] || left.index - right.index)
  return { positions: valid[0].positions, actionIsSet: true }
}

async function main() {
  const [evidenceArg, selectedArg, reportArg, extra] = process.argv.slice(2)
  if (!evidenceArg || !selectedArg || !reportArg || extra) {
    throw new Error('usage: node tools/analyze-active-v15-readers.mjs EVIDENCE_JSONL SELECTED_JSON REPORT_JSON')
  }
  const records = (await readFile(resolve(evidenceArg), 'utf8')).split('\n').filter(Boolean).map(JSON.parse)
  if (records.length !== 1_024) throw new Error(`expected 1024 evidence records, got ${records.length}`)
  for (const record of records) {
    validateRecord(record)
    record.exactScores = exactScores(record)
  }
  const { fitting, validation } = splitRecords(records)
  const currentAction = (record) => ({ positions: record.strongestPositions, actionIsSet: record.actionIsSet })
  const readers = [{
    name: 'current',
    fitting: evaluate(fitting, currentAction),
    validation: evaluate(validation, currentAction),
  }]
  readers.push(fitThreshold('exact-continuous', fitting, validation, (record) => record.exactScores))

  const learnedMoments = scoreMoments(fitting, (record) => record.tripleScores)
  const exactMoments = scoreMoments(fitting, (record) => record.exactScores)
  for (const learnedWeight of [0.25, 0.5, 0.75]) {
    const score = (record) => record.tripleScores.map((learned, index) =>
      learnedWeight * (learned - learnedMoments.mean) / learnedMoments.stddev
      + (1 - learnedWeight) * (record.exactScores[index] - exactMoments.mean) / exactMoments.stddev)
    readers.push(fitThreshold(
      `blend-${String(Math.round(learnedWeight * 100)).padStart(3, '0')}`,
      fitting,
      validation,
      score,
      { learnedWeight, learnedMoments, exactMoments },
    ))
  }
  readers.push({
    name: 'discrete-attributes',
    fitting: evaluate(fitting, discreteAction),
    validation: evaluate(validation, discreteAction),
  })

  const priority = ['current', 'exact-continuous', 'blend-050', 'blend-025', 'blend-075', 'discrete-attributes']
  readers.sort((left, right) => left.validation.failures - right.validation.failures
    || right.validation.worstConditional - left.validation.worstConditional
    || priority.indexOf(left.name) - priority.indexOf(right.name))
  const selected = readers[0]
  const config = {
    type: selected.name,
    ...(selected.threshold === undefined ? {} : { threshold: selected.threshold }),
    ...(selected.learnedWeight === undefined ? {} : {
      learnedWeight: selected.learnedWeight,
      learnedMoments: selected.learnedMoments,
      exactMoments: selected.exactMoments,
    }),
  }
  await writeFile(resolve(selectedArg), `${JSON.stringify(config, null, 2)}\n`, { flag: 'wx' })
  await writeFile(resolve(reportArg), `${JSON.stringify({
    protocol: 'active-v15', fittingRecords: fitting.length, validationRecords: validation.length,
    selected: config, readers,
  }, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify({ selected: config, fitting: selected.fitting, validation: selected.validation }))
}

main().catch((error) => {
  console.error(error?.stack ?? String(error))
  process.exitCode = 1
})
