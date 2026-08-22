#!/usr/bin/env node

import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import { boardActionIsSet, validateBoardHead } from '../extension/board-head.js'
import { evaluate, loadRecords } from './fit-active-v19-head.mjs'

const [evidenceRootArg, headArg, reportArg, extra] = process.argv.slice(2)
if (!evidenceRootArg || !headArg || !reportArg || extra) {
  throw new Error('usage: verify-active-v26-runtime EVIDENCE_ROOT HEAD REPORT')
}

const records = await loadRecords(resolve(evidenceRootArg))
const head = validateBoardHead(JSON.parse(await readFile(resolve(headArg), 'utf8')))
const report = JSON.parse(await readFile(resolve(reportArg), 'utf8'))
const action = (record) => boardActionIsSet({
  ...record,
  tripleScores: record.sortedTripleScores,
}, head)
const actual = {
  fitting: evaluate(records.slice(0, 8192), action),
  validation: evaluate(records.slice(8192), action),
}
for (const split of ['fitting', 'validation']) {
  const expected = report.candidate[split]
  if (JSON.stringify(actual[split]) !== JSON.stringify(expected)) {
    throw new Error(`${split} runtime metrics do not reproduce the fitter report`)
  }
}
console.log(JSON.stringify(actual))
