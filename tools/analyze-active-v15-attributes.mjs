#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const ATTRIBUTE_NAMES = ['count', 'colour', 'shape', 'fill']
const CARD_COUNT = 12
const ATTRIBUTE_VALUES = 3

function cardAttributes(id) {
  return [Math.floor(id / 27) % 3, Math.floor(id / 9) % 3, Math.floor(id / 3) % 3, id % 3]
}

function emptySummary() {
  return {
    cards: 0,
    attributes: ATTRIBUTE_NAMES.map((name) => ({
      name,
      correct: 0,
      total: 0,
      confusion: Array.from({ length: ATTRIBUTE_VALUES }, () => Array(ATTRIBUTE_VALUES).fill(0)),
    })),
  }
}

function observe(summary, record) {
  if (!Array.isArray(record.boardIds) || record.boardIds.length !== CARD_COUNT
    || !Array.isArray(record.attributeProbabilities)
    || record.attributeProbabilities.length !== CARD_COUNT * ATTRIBUTE_NAMES.length * ATTRIBUTE_VALUES) {
    throw new Error(`iteration ${record.iteration}: incomplete attribute evidence`)
  }
  summary.cards += CARD_COUNT
  for (let card = 0; card < CARD_COUNT; card++) {
    const expected = cardAttributes(record.boardIds[card])
    for (let attribute = 0; attribute < ATTRIBUTE_NAMES.length; attribute++) {
      const start = (card * ATTRIBUTE_NAMES.length + attribute) * ATTRIBUTE_VALUES
      const probabilities = record.attributeProbabilities.slice(start, start + ATTRIBUTE_VALUES)
      const predicted = probabilities.indexOf(Math.max(...probabilities))
      const metric = summary.attributes[attribute]
      metric.total++
      metric.correct += Number(predicted === expected[attribute])
      metric.confusion[expected[attribute]][predicted]++
    }
  }
}

function finalize(summary) {
  for (const attribute of summary.attributes) attribute.accuracy = attribute.correct / attribute.total
  return summary
}

async function main() {
  const [evidenceArg, outputArg, extra] = process.argv.slice(2)
  if (!evidenceArg || !outputArg || extra) {
    throw new Error('usage: node tools/analyze-active-v15-attributes.mjs EVIDENCE_JSONL OUTPUT_JSON')
  }
  const records = (await readFile(resolve(evidenceArg), 'utf8')).split('\n').filter(Boolean).map(JSON.parse)
  const overall = emptySummary()
  const profiles = {}
  for (const record of records) {
    profiles[record.profile] ??= emptySummary()
    observe(overall, record)
    observe(profiles[record.profile], record)
  }
  finalize(overall)
  for (const profile of Object.values(profiles)) finalize(profile)
  const report = { boards: records.length, overall, profiles }
  await writeFile(resolve(outputArg), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify({
    overall: Object.fromEntries(overall.attributes.map((attribute) => [attribute.name, attribute.accuracy])),
    crop: Object.fromEntries(profiles.crop.attributes.map((attribute) => [attribute.name, attribute.accuracy])),
  }))
}

main().catch((error) => {
  console.error(error?.stack ?? String(error))
  process.exitCode = 1
})
