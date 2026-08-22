#!/usr/bin/env node

import { readFile, readdir } from 'node:fs/promises'
import { resolve } from 'node:path'

const [referenceDirectoryArgument, candidateDirectoryArgument, boardsArgument] = process.argv.slice(2)
const boards = Number(boardsArgument)
if (!referenceDirectoryArgument || !candidateDirectoryArgument
  || !Number.isSafeInteger(boards) || boards <= 0) {
  throw new Error('usage: node tools/compare-active-v29-paired.mjs REFERENCE_DIR CANDIDATE_DIR BOARDS')
}

async function failureIndicators(directoryArgument) {
  const directory = resolve(directoryArgument)
  const files = (await readdir(directory)).filter((file) => file.endsWith('.log')).sort()
  const failures = new Uint8Array(boards)
  for (const file of files) {
    const lines = (await readFile(resolve(directory, file), 'utf8')).split('\n')
    for (const line of lines) {
      let record
      try { record = JSON.parse(line) } catch { continue }
      if (record.event !== 'failure') continue
      if (!Number.isSafeInteger(record.iteration) || record.iteration < 0 || record.iteration >= boards) {
        throw new Error(`${file}: failure iteration outside 0..${boards - 1}`)
      }
      failures[record.iteration] = 1
    }
  }
  return failures
}

function nextRandom(state) {
  state.value ^= state.value << 13
  state.value ^= state.value >>> 17
  state.value ^= state.value << 5
  state.value >>>= 0
  return state.value
}

const reference = await failureIndicators(referenceDirectoryArgument)
const candidate = await failureIndicators(candidateDirectoryArgument)
const differences = Int8Array.from(reference, (failure, index) => failure - candidate[index])
const improvement = differences.reduce((total, value) => total + value, 0)
const bootstrapReplicates = 20_000
const bootstrap = new Float64Array(bootstrapReplicates)
const randomState = { value: 2026090892 }
for (let replicate = 0; replicate < bootstrapReplicates; replicate++) {
  let total = 0
  for (let sample = 0; sample < boards; sample++) {
    total += differences[nextRandom(randomState) % boards]
  }
  bootstrap[replicate] = total
}
const bootstrapMean = bootstrap.reduce((total, value) => total + value, 0) / bootstrapReplicates
const bootstrapVariance = bootstrap.reduce(
  (total, value) => total + (value - bootstrapMean) ** 2,
  0,
) / (bootstrapReplicates - 1)
const standardError = Math.sqrt(bootstrapVariance)
const report = {
  boards,
  bootstrapReplicates,
  bootstrapSeed: 2026090892,
  referenceFailures: reference.reduce((total, value) => total + value, 0),
  candidateFailures: candidate.reduce((total, value) => total + value, 0),
  pairedImprovement: improvement,
  pairedStandardError: standardError,
  requiredImprovement: 2 * standardError,
  passesTwoStandardErrors: improvement >= 2 * standardError,
}
console.log(JSON.stringify(report, null, 2))
