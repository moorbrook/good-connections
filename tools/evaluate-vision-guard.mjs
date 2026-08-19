import { readFile } from 'node:fs/promises'
import { diagnosticsPass } from '../extension/guard.js'

const [samplesPath] = process.argv.slice(2)
if (!samplesPath) {
  console.error('Usage: node tools/evaluate-vision-guard.mjs ABSTENTION_SAMPLES.tsv')
  process.exit(2)
}

const rows = (await readFile(samplesPath, 'utf8')).trim().split('\n')
const header = rows.shift()?.split('\t') ?? []
const column = Object.fromEntries(header.map((name, index) => [name, index]))
const groups = new Map()
const overall = { boards: 0, errors: 0, accepted: 0, acceptedErrors: 0 }

for (const row of rows) {
  const values = row.split('\t')
  const atlas = values[column.atlas]
  const profile = atlas.match(/-p(blur|crop|shadow|lowcontrast|jpeg|defocus|tilt|occlusion|gamut|compound)/)?.[1] ?? 'clean'
  const correct = values[column.correct] === '1'
  const strongestIsSet = values[column.strongest_is_set] === '1'
  const observation = {
    score: Number(values[column.score]),
    scoreMargin: Number(values[column.score_margin]),
    attributeEntropy: Number(values[column.mean_attribute_entropy]),
    attributeMargin: Number(values[column.min_attribute_margin]),
  }
  const accepted = strongestIsSet && diagnosticsPass(observation)
  const group = groups.get(profile) ?? { boards: 0, errors: 0, accepted: 0, acceptedErrors: 0 }
  for (const stats of [overall, group]) {
    stats.boards++
    stats.errors += Number(!correct)
    stats.accepted += Number(accepted)
    stats.acceptedErrors += Number(accepted && !correct)
  }
  groups.set(profile, group)
}

const summarize = (stats) => ({
  ...stats,
  rawAccuracy: 1 - stats.errors / stats.boards,
  coverage: stats.accepted / stats.boards,
  acceptedAccuracy: stats.accepted === 0 ? null : 1 - stats.acceptedErrors / stats.accepted,
})
console.log(JSON.stringify({
  samples: samplesPath,
  overall: summarize(overall),
  profiles: Object.fromEntries([...groups].sort().map(([name, stats]) => [name, summarize(stats)])),
}, null, 2))
